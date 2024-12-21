import yaml
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
import torch.optim as optim
from torch.cuda.amp import GradScaler

from data.data_loader import DataLoader as CustomDataLoader
from data.dataset import ProductReviewDataset
from models.model import ImprovedProductClassifier
from models.loss import FocalLoss
from trainer.trainer import ModelTrainer
from inference.pipeline import InferencePipeline
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

nltk.download('stopwords')
nltk.download('wordnet')

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_model(config, n_classes):
    """Initialize model and training components"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ImprovedProductClassifier(
        n_classes=n_classes,
        model_name=config['model']['name'],
        dropout_rate=float(config['model']['dropout_rate'])
    )
    model = model.to(device)
    
    # Initialize optimizer with parameter groups
    no_decay = ['bias', 'LayerNorm.weight']
    base_lr = float(config['training']['learning_rate'])  # Ensure float conversion
    weight_decay = float(config['training']['weight_decay'])  # Ensure float conversion
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' in n],
            'lr': base_lr,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' in n],
            'lr': base_lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' not in n],
            'lr': base_lr * 10,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' not in n],
            'lr': base_lr * 10,
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    criterion = FocalLoss(alpha=1.5, gamma=2)
    
    return model, optimizer, criterion, device

def main():
    try:
        # Load configuration
        config = load_config('config/config.yml')
        logger.info("Configuration loaded successfully")
        
        # Load data
        data_loader = CustomDataLoader(config)
        data = data_loader.load_data()
        logger.info("Data loaded successfully")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        # Create datasets
        train_dataset = ProductReviewDataset(
            data['train_texts'],
            data['train_labels'],
            tokenizer,
            max_len=int(config['model']['max_len']),  # Ensure int conversion
            augment=True
        )
        
        val_dataset = ProductReviewDataset(
            data['val_texts'],
            data['val_labels'],
            tokenizer,
            max_len=int(config['model']['max_len']),  
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config['training']['batch_size']),  
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config['training']['batch_size']),  
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info("DataLoaders created successfully")
        
        # Setup model and training components
        model, optimizer, criterion, device = setup_model(config, len(data['class_names']))
        
        # Create scheduler with proper type conversion
        total_steps = len(train_loader) * int(config['training']['epochs'])
        base_lr = float(config['training']['learning_rate'])
        max_lrs = [base_lr] * 2 + [base_lr * 10] * 2  
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_names=data['class_names']
        )
        
        # Train model
        history = trainer.train(
            num_epochs=int(config['training']['epochs']),  
            patience=int(config['training']['patience']), 
            checkpoint_path='best_model.pt'
        )
        
        
        # Load best model and create inference pipeline
        pipeline = InferencePipeline('best_model.pt')
        
        # Test predictions
        test_texts = [
            "The camera quality is excellent with great detail in low light conditions",
            "Battery life is disappointing, barely lasts half a day",
            "The build quality is premium with a solid metal frame"
        ]
        
        logger.info("\nTesting predictions:")
        results = pipeline.predict_batch(test_texts)
        
        for result in results:
            print(f"\nInput: {result['text']}")
            print("Predictions:")
            for pred in result['predictions']:
                if pred['predicted']:
                    print(f"{pred['category']}: {pred['confidence']:.4f}")
        
        logger.info("Training and evaluation completed successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    pipeline = main()