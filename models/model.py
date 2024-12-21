import torch
from torch import nn
from transformers import AutoModel

class ImprovedProductClassifier(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-uncased', dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze the first 8 layers of BERT
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        self.attention = nn.MultiheadAttention(768, 8, dropout=dropout_rate)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
        
        self.layer_norm = nn.LayerNorm(768)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        sequence_output = self.layer_norm(outputs.last_hidden_state)
        
        attention_output, _ = self.attention(
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2)
        )
        attention_output = attention_output.permute(1, 0, 2)
        
        lstm_output, _ = self.lstm(attention_output)
        
        avg_pooled = self.avg_pool(lstm_output.transpose(1, 2)).squeeze(-1)
        max_pooled = self.max_pool(lstm_output.transpose(1, 2)).squeeze(-1)
        
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        return self.classifier(pooled)
