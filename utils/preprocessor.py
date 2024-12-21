import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_text(self, text, augment=False):
        try:
            text = self.clean_text(text)
            if not text:
                return ""
            
            words = text.split()
            words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words and len(word) > 2
            ]
            
            return ' '.join(words)
            
        except Exception as e:
            return text