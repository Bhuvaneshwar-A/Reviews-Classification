import torch
from torch.utils.data import Dataset
from utils.preprocessor import TextPreprocessor
import random

class ProductReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.preprocessor = TextPreprocessor()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.augment and random.random() < 0.3:
            text = self.preprocessor.preprocess_text(text, augment=True)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }