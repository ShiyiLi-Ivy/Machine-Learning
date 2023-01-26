import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bert, hidden_size=32, num_of_class=2, dropout=0.1):
        super(Model, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_of_class)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
