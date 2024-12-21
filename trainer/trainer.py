import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, class_names):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_names = class_names
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        self.best_val_f1 = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast():
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {}
        pred_labels = (all_preds > 0.5).astype(int)
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[:, i],
                pred_labels[:, i],
                average='binary'
            )
            
            try:
                roc_auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except:
                roc_auc = 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
        
        return total_loss / len(self.val_loader), metrics, all_preds, all_labels
    
    def train(self, num_epochs, patience=5, checkpoint_path='best_model.pt'):
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            val_loss, metrics, predictions, labels = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)
            
            avg_f1 = np.mean([m['f1'] for m in metrics.values()])
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Average F1: {avg_f1:.4f}")
            
            if avg_f1 > self.best_val_f1:
                self.best_val_f1 = avg_f1
                self.save_checkpoint(checkpoint_path, epoch)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break
        
        return self.history
    
    def save_checkpoint(self, path, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'class_names': self.class_names
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint