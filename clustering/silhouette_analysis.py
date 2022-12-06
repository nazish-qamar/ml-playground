import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


class SilhouetteAnalysis:
    def __init__(self):
        self.cluster_labels = None
        self.n_clusters = None

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
            km = KMeans(n_clusters=3,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        tol=1e-04,
                        random_state=0)
            y_km = km.fit_predict(X)
            self.cluster_labels = np.unique(y_km)
            self.n_clusters = self.cluster_labels.shape[0]
            return y_km

    def plot_silhouette(self, X, y_km):
        silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(self.cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / self.n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                     edgecolor='none', color=color)

            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)

        #adding average silhouette coefficient
        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--")

        plt.yticks(yticks, self.cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette coefficient')

        plt.tight_layout()
        plt.show()

    def run(self):
        X = self.create_dummy_data()
        y_km = self.kmeans_clustering(X)
        self.plot_silhouette(X, y_km)


SilhouetteAnalysis().run()