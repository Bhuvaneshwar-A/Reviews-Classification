import torch
from transformers import AutoTokenizer
import numpy as np
from utils.preprocessor import TextPreprocessor
from models.model import ImprovedProductClassifier
import logging

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self, model_path='best_model.pt', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        self.preprocessor = TextPreprocessor()
        logger.info(f"Inference pipeline initialized on device: {self.device}")
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = ImprovedProductClassifier(n_classes=len(checkpoint['class_names']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.class_names = checkpoint['class_names']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, text):
        """Preprocess input text"""
        processed_text = self.preprocessor.preprocess_text(text)
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
            'token_type_ids': encoding['token_type_ids'].to(self.device)
        }
    
    def predict(self, text, threshold=0.5):
        """Make predictions for a single text input"""
        try:
            # Preprocess input
            inputs = self.preprocess_input(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Format results
            results = []
            for class_name, prob in zip(self.class_names, probabilities):
                confidence = float(prob)
                is_predicted = confidence >= threshold
                results.append({
                    'category': class_name,
                    'confidence': confidence,
                    'predicted': is_predicted
                })
            
            return sorted(results, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def predict_batch(self, texts, threshold=0.5, batch_size=16):
        """Make predictions for a batch of texts"""
        try:
            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = []
                
                # Process each text in the batch
                for text in batch_texts:
                    prediction = self.predict(text, threshold)
                    batch_results.append({
                        'text': text,
                        'predictions': prediction
                    })
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    def analyze_prediction(self, text, threshold=0.5):
        """Provide detailed analysis of prediction"""
        try:
            # Get predictions
            predictions = self.predict(text, threshold)
            processed_text = self.preprocessor.preprocess_text(text)
            
            analysis = {
                'original_text': text,
                'processed_text': processed_text,
                'predictions': predictions,
                'summary': {
                    'n_categories': sum(1 for p in predictions if p['predicted']),
                    'avg_confidence': np.mean([p['confidence'] for p in predictions]),
                    'max_confidence': max(p['confidence'] for p in predictions),
                    'min_confidence': min(p['confidence'] for p in predictions)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in prediction analysis: {e}")
            raise