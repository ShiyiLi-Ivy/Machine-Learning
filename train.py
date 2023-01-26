import numpy as np
import torch
import torch.nn as nn
from torch import optim


def train(model, train_dataloader, val_dataloader, learning_rate, num_epochs, device):
    # Start model training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fun = nn.CrossEntropyLoss()
    loss_fun = loss_fun.to(device)

    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, num_epochs))

        model.train()
        total_loss, total_acc, total_size = 0, 0, 0

        for i, batch in enumerate(train_dataloader):
            batch = [r.to(device) for r in batch]  # push the batch to device
            sent_id, mask, y = batch

            logits = model(sent_id, mask)
            optimizer.zero_grad()
            loss = loss_fun(logits, y)
            loss.backward()
            optimizer.step()

            logits = logits.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pre = np.argmax(logits, axis=1)
            total_loss += loss.item()
            total_acc += (y_pre == y).sum().item()
            total_size += len(y_pre)

        train_loss = total_loss / total_size
        train_acc = total_acc / total_size
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()  # deactivate dropout layers
        total_loss, total_acc, total_size = 0, 0, 0

        for i, batch in enumerate(val_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, y = batch
            with torch.no_grad():
                logits = model(sent_id, mask)
                loss = loss_fun(logits, y)

                logits = logits.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                y_pre = np.argmax(logits, axis=1)
                total_loss += loss.item()
                total_acc += (y_pre == y).sum().item()
                total_size += len(y_pre)

        val_loss = total_loss / total_size
        val_acc = total_acc / total_size
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'saved_weights.pt')

        print(f'\nTraining Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'\nTraining Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')

    return train_losses, train_accs, val_losses, val_accs
