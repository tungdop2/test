import torch
from torch import nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")

#         for param in self.bert.parameters():
#             param.requires_grad = False

        self.drop = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )
        x = output
        x = self.drop(x)
        x = self.fc(x)
        return x