from sklearn.neighbors import KNeighborsClassifier
from load_data import LoadData


class KNN:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.knn_clf = None

    def model_fit(self):
        self.knn_clf = KNeighborsClassifier(n_neighbors=5,
                                            p=2, #p=2 for Euclidean distance
                                            metric='minkowski')
        self.knn_clf.fit(self.X_train, self.y_train)

    def predict_(self):
        print("Predictions: ", self.knn_clf.predict(self.X_test))
        print("True Labels: ", self.y_test)


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True)
knn_ = KNN(X_train, y_train, X_test, y_test)
knn_.model_fit()
knn_.predict_()
