import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


class ElbowMethod:
    def __init__(self):
        self.distortions = []

    def create_dummy_data(self):
        X, y = make_blobs(n_samples=150,
                          n_features=2,
                          centers=3,
                          cluster_std=0.5,
                          shuffle=True,
                          random_state=0)
        return X

    def kmeans_clustering(self, X):
        for i in range(1, 11):
            km = KMeans(n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)
            km.fit(X)
            self.distortions.append(km.inertia_)

    def plot_distortions(self):
        plt.plot(range(1, 11), self.distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()
        plt.show()

    def run(self):
        X = self.create_dummy_data()
        self.kmeans_clustering(X)
        self.plot_distortions()


ElbowMethod().run()