import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


class DBSCANClustering:
    def __init__(self):
        self.X = None

    def create_data(self):
        self.X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    def find_clusters(self):
        db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
        y_db = db.fit_predict(self.X)
        plt.scatter(self.X[y_db == 0, 0], self.X[y_db == 0, 1],
                    c='lightblue', marker='o', s=40,
                    edgecolor='black',
                    label='Cluster 1')
        plt.scatter(self.X[y_db == 1, 0], self.X[y_db == 1, 1],
                    c='red', marker='s', s=40,
                    edgecolor='black',
                    label='Cluster 2')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.create_data()
        self.find_clusters()


DBSCANClustering().run()