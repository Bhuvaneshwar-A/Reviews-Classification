import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mlb = MultiLabelBinarizer()

    def load_data(self):
        # Load training data
        train_df = pd.read_csv(self.config['data']['train_file'])
        
        # Clean category strings
        train_df['Category'] = train_df['Category'].apply(
            lambda x: ast.literal_eval(x) if pd.notnull(x) else []
        )
        
        # Combine text columns
        train_df['combined_text'] = train_df.apply(
            lambda x: f"{x['reviews']} {x['product_description']}", 
            axis=1
        )
        
        # Transform labels
        labels = self.mlb.fit_transform(train_df['Category'])
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_df['combined_text'].values,
            labels,
            test_size=self.config['data']['validation_split'],
            random_state=42,
            stratify=labels.sum(axis=1)
        )
        
        return {
            'train_texts': train_texts,
            'val_texts': val_texts,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'class_names': self.mlb.classes_
        }