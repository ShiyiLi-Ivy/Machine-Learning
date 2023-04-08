import numpy as np
from matplotlib import pyplot as plt


def cecost(y, yhat):
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))


def cecostprime(yhat, y):
    return -((y / yhat) - ((1 - y) / (1 - yhat)))


def sigmoid(a):
    return 1. / (1. + np.exp(-a))


def sigmoid_prime(a):
    return sigmoid(a) * (1. - sigmoid(a))


def ReLU(a):
    return 0 if a < 0 else a


def ReLU_prime(a):
    return 1 if a > 0 else 0


def plot(train_accs, val_accs, num_epochs):
    x = list(range(1, num_epochs + 1))
    plt.plot(x, train_accs, label='train', color='red')
    plt.plot(x, val_accs, label='val', color='green')
    plt.legend(fontsize=10)
    plt.title("Train/Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()
