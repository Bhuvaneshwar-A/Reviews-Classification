# Product Classification Model

A deep learning model for multi-label product classification using BERT and custom architecture.

## Features
- Multi-label product classification
- BERT-based architecture with custom improvements
- Text preprocessing and augmentation
- Training with mixed precision and gradient accumulation
- Flexible inference pipeline

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the model:
   - Edit `config/config.yml` to set model parameters, training settings, etc.

2. Prepare your data:
   - Place your training data in CSV format
   - Required columns: 'reviews', 'Category'
   - Categories should be in list format

3. Train the model:
```bash
python main.py
```

4. Use the inference pipeline:
```python
from inference.pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline('best_model.pt')

# Make predictions
text = "Your product review or description here"
predictions = pipeline.predict(text)

# Batch predictions
texts = ["Review 1", "Review 2", "Review 3"]
batch_predictions = pipeline.predict_batch(texts)
```

## Model Architecture

The model uses a BERT base with several improvements:
- Frozen first 8 layers for efficiency
- Multi-head attention layer
- Bidirectional LSTM
- Advanced pooling strategies
- Custom focal loss for handling class imbalance

## Training Features

- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Comprehensive metrics tracking

## Acknowledgments

- Hugging Face Transformers library
- PyTorch
- NLTK
- scikit-learn