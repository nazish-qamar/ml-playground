from sklearn.ensemble import RandomForestClassifier
from load_data import LoadData


class RandomForest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.forest = None

    def model_fit(self):
        self.forest = RandomForestClassifier(criterion='gini',
                                             n_estimators=25,
                                             random_state=1,
                                             n_jobs=2) #for parallelizing model training over 2 cores
        self.forest.fit(self.X_train, self.y_train)

    def predict_(self):
        print("Predictions: ", self.forest.predict(self.X_test))
        print("True Labels: ", self.y_test)


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=False)
forest = RandomForest(X_train, y_train, X_test, y_test)
forest.model_fit()
forest.predict_()
