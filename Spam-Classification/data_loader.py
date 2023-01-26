import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from transformers import BertTokenizerFast


class TextDataset(Dataset):
    def __init__(self, seq, mask, label):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long)
        self.label = torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        return self.seq[index], self.mask[index], self.label[index]

    def __len__(self):
        return len(self.seq)


def make_dataloader(X, y, max_length=100, batch_size=32):
    # Split train dataset into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, random_state=2023, test_size=0.4, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, random_state=2023, test_size=0.5, stratify=y_temp)

    # Tokenization
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(
        X_train, max_length=max_length, padding=True, truncation=True, return_token_type_ids=False)
    tokens_val = tokenizer.batch_encode_plus(
        X_val, max_length=max_length, padding=True, truncation=True, return_token_type_ids=False)
    tokens_test = tokenizer.batch_encode_plus(
        X_test, max_length=max_length, padding=True, truncation=True, return_token_type_ids=False)

    train_dataset = TextDataset(
        tokens_train['input_ids'], tokens_train['attention_mask'], y_train)
    val_dataset = TextDataset(
        tokens_val['input_ids'], tokens_val['attention_mask'], y_val)
    test_dataset = TextDataset(
        tokens_test['input_ids'], tokens_test['attention_mask'], y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    # Example for BERT Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    text = ["this is a bert model tutorial", "we will fine-tune a bert model"]
    sent_id = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False)
    print(sent_id)
