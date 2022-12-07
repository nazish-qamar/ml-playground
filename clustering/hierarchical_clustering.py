import numpy as np
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClustering:
    def __init__(self):
        self.X = None

    def create_data(self):
        np.random.seed(123)
        self.X = np.random.random_sample([5, 3])*10
        print(self.X)

    def find_clusters(self):
        ac = AgglomerativeClustering(n_clusters=3,
                                     affinity='euclidean',
                                     linkage='complete')
        labels = ac.fit_predict(self.X)
        print('Cluster labels: %s' % labels)

    def run(self):
        self.create_data()
        self.find_clusters()


HierarchicalClustering().run()