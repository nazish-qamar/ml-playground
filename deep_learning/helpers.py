import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

np.random.seed(1)

def generate_xor_data():
    x = np.random.uniform(low=-1, high=1, size=(200, 2))
    y = np.ones(len(x))
    y[x[:, 0] * x[:, 1] < 0] = 0

    x_train = x[:100, :]
    y_train = y[:100]
    x_valid = x[100:, :]
    y_valid = y[100:]
    return x_train, y_train, x_valid, y_valid


def plot_performance(model, history, X_val, y_val):
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history['loss'], lw=4)
    plt.plot(history['val_loss'], lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history['binary_accuracy'], lw=4)
    plt.plot(history['val_binary_accuracy'], lw=4)
    plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 3)
    plot_decision_regions(X=X_val, y=y_val.astype(np.int32),
                          clf=model)
    ax.set_xlabel(r'$x_1$', size=15)
    ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(-0.025, 1)
    plt.show()