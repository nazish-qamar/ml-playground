from load_data import LoadData
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def model_fit(self):
        self.model = DecisionTreeClassifier(criterion='entropy',
                                            max_depth=4,
                                            random_state=1)
        self.model.fit(self.X_train, self.y_train)

    def predict_(self):
        print("Predictions: ", self.model.predict(self.X_test))
        print("True Labels: ", self.y_test)


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=False)
model = DecisionTree(X_train, y_train, X_test, y_test)
model.model_fit()
model.predict_()
