import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class LoadMnist:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.y_test = None
        self.X_test = None

    def load_train_test(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype(int)
        X = ((X / 255.) - .5) * 2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=10000,
                                                                                random_state=123,
                                                                                stratify=y)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def visualize(self):
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(10):
            img = self.X_train.iloc[[i],].to_numpy().reshape(28, 28)
            ax[i].imshow(img, cmap='Greys')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()

    def save_compressed_data(self):
        np.savez_compressed('mnist_scaled.npz',
                            X_train=self.X_train,
                            y_train=self.y_train,
                            X_test=self.X_test,
                            y_test=self.y_test)

    def load_uncompressed_data(self):
        mnist = np.load('mnist_scaled.npz')
        #print(mnist.files)
        X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train',
                                                               'y_train',
                                                               'X_test',
                                                               'y_test']]

        del mnist
        return X_train, y_train, X_test, y_test


#model_ = LoadMnist()
#X_train, X_test, y_train, y_test = model_.load_train_test()
#model_.save_compressed_data()