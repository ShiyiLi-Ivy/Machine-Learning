import torch
from transformers import AutoModel

from preprocess import load_cleaned_data
from data_loader import make_dataloader
from model import Model
from train import train
from evaluate import evaluate
from utils import plot


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_cleaned_data()

    # Import BERT Model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    # Freeze BERT parameters
    for param in bert.parameters():
        param.requires_grad = False

    train_dataloader, val_dataloader, test_dataloader = make_dataloader(X, y, max_length=200, batch_size=32)

    # Pass the pre-trained BERT to our define architecture
    model = Model(bert)
    model = model.to(device)

    learning_rate = 1e-3
    num_epochs = 5

    train_losses, train_accs, val_losses, val_accs = train(
        model, train_dataloader, val_dataloader, learning_rate, num_epochs, device)
    plot(train_losses, train_accs, val_losses, val_accs, num_epochs)

    evaluate(model, test_dataloader, device)
