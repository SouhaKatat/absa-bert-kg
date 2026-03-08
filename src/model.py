
import torch
from torch import nn


class BERT_ABSA(nn.Module):

    def __init__(self, bert_model, num_classes=3):

        super().__init__()

        self.bert = bert_model

        self.dropout = nn.Dropout(0.1)

        self.dimension_adjustment = nn.Linear(768 + 300, 1024)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=6
        )

        self.attention = nn.MultiheadAttention(
            1024,
            num_heads=8,
            batch_first=True
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input_ids, attention_mask, node_embeddings):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs[1]

        pooled = self.dropout(pooled)

        combined = torch.cat((pooled, node_embeddings), dim=1)

        combined = self.dimension_adjustment(combined)

        attention_output, _ = self.attention(
            combined.unsqueeze(1),
            combined.unsqueeze(1),
            combined.unsqueeze(1)
        )

        combined = attention_output.squeeze(1) + combined

        encoded = self.transformer_encoder(combined.unsqueeze(0))

        logits = self.classifier(encoded.squeeze(0))

        return logits
