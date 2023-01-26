import numpy as np
import torch
from sklearn.metrics import classification_report


def evaluate(model, test_dataloader, device):
    # Load weights of the best model
    model.load_state_dict(torch.load('saved_weights.pt'))
    # Get predictions for test data
    preds = []
    labels = []
    for i, batch in enumerate(test_dataloader):
        batch = [r.to(device) for r in batch]  # push the batch to device
        sent_id, mask, y = batch
        with torch.no_grad():
            logits = model(sent_id, mask)
            logits = logits.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pre = np.argmax(logits, axis=1)
            preds.append(y_pre)
            labels.append(y)

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(classification_report(labels, preds))
