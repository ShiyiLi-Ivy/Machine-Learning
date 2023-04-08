import numpy as np
import pickle
import gzip

from network import Network
from utils import plot

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    is_train = False

    with gzip.open("./mnist.pkl.gz", "rb") as file:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = pickle.load(file, encoding="latin-1")

    # one-hot-encode the labels
    y_train = [[0] * (elt) + [1] + [0] * (9 - elt) for elt in y_train]
    y_val = [[0] * (elt) + [1] + [0] * (9 - elt) for elt in y_val]
    y_test = [[0] * (elt) + [1] + [0] * (9 - elt) for elt in y_test]

    epochs = 10  # default: 10
    lr = 0.15  # default: 0.15
    batch_size = 128  # default: 128
    hidden_size = 100  # default: 100
    weight_decay = 1e-3  # default: 1e-3
    lr_decay = 0.1  # default: 0.1

    nn = Network(784, hidden_size, 10, None)
    if is_train:
        params, train_accs, val_accs = nn.learn(
            X_train, y_train, lr, epochs, batch_size, X_val, y_val, weight_decay, lr_decay)
        plot(train_accs, val_accs, epochs)
        np.savez("./model_params.npz", **params)
    else:
        params = np.load("./model_params.npz", allow_pickle=True)
        nn.load_and_test(params, X_test, y_test)
