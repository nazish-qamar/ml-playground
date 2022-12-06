import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from helper.load_data import LoadData


class KMeansClustering:
    def __init__(self):
        self.y_km = None
        self.km = KMeans(n_clusters=2,
                         init='random', # use init = 'k-means++' for kmeans++ clustering (by default init is k-means++)
                         n_init=10,
                         max_iter=300,
                         tol=1e-04,
                         random_state=0)

    def create_cluster(self, X):
        X["Cluster"] = self.km.fit_predict(X)
        X["Cluster"] = X["Cluster"].astype("category")
        sns.relplot(
            x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
        )
        plt.show()
        return self


df = LoadData().load_california_housing_data()
model_ = KMeansClustering()
model_.create_cluster(df)
