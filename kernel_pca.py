import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.datasets import make_moons


class Rbf_Pca:
    def __init__(self, X, y, gamma, n_components):
        self.X = X
        self.y = y
        self.gamma = gamma
        self.n_components = n_components
        self.X_pc = None

    def view_original_data(self):
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
        plt.tight_layout()
        plt.title("Original Data")
        plt.show()

    def rbf_kernel_pca(self):
        sq_dists = pdist(self.X, 'sqeuclidean')     # Euclidean distances
        mat_sq_dists = squareform(sq_dists) # # Convert into a square matrix.
        K = np.exp(-self.gamma * mat_sq_dists)     # Compute the symmetric kernel matrix.

        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Obtaining eigenpairs from the centered kernel matrix
        # scipy.linalg.eigh returns them in ascending order
        eigvals, eigvecs = eigh(K)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

        # Collect the top k eigenvectors (projected examples)
        self.X_pc = np.column_stack([eigvecs[:, i] for i in range(self.n_components)])

    def plot_pca(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
        ax[0].scatter(self.X_pc[y == 0, 0], self.X_pc[y == 0, 1],
                      color='red', marker='^', alpha=0.5)
        ax[0].scatter(self.X_pc[y == 1, 0], self.X_pc[y == 1, 1],
                      color='blue', marker='o', alpha=0.5)

        ax[1].scatter(self.X_pc[y == 0, 0], np.zeros((50, 1)) + 0.02,
                      color='red', marker='^', alpha=0.5)
        ax[1].scatter(self.X_pc[y == 1, 0], np.zeros((50, 1)) - 0.02,
                      color='blue', marker='o', alpha=0.5)

        ax[0].set_xlabel('PC1')
        ax[0].set_ylabel('PC2')
        ax[0].set_title("Data after kernel PCA transformation")
        ax[1].set_ylim([-1, 1])
        ax[1].set_yticks([])
        ax[1].set_xlabel('PC1')
        ax[1].set_title("Projecting on X axis")

        plt.tight_layout()
        plt.show()


X, y = make_moons(n_samples=100, random_state=123)
rbf_obj = Rbf_Pca(X,y,gamma=15, n_components=2)
rbf_obj.view_original_data()
rbf_obj.rbf_kernel_pca()
rbf_obj.plot_pca()

