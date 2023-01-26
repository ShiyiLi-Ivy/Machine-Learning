from matplotlib import pyplot as plt


def plot(train_losses, train_accs, val_losses, val_accs, num_epochs):
    x = list(range(1, num_epochs + 1))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='train', color='red')
    plt.plot(x, val_losses, label='val', color='green')
    plt.legend(fontsize=10)
    plt.title("Train/Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(x, train_accs, label='train', color='red')
    plt.plot(x, val_accs, label='val', color='green')
    plt.legend(fontsize=10)
    plt.title("Train/Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, 4, forward=True)
    plt.show()
