import numpy as np
from helper.load_data import LoadData


class LogisticRegressionWithGD:
    def __init__(self, eta=0.05, epochs=100, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.randomnum_generator = np.random.RandomState(self.random_state)
        self.weights = 0
        self.cost = []

    def fit(self, X, y):
        self.weights = self.randomnum_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost = []

        for i in range(self.epochs):
            output = self.activation(np.dot(X, self.weights[1:]) + self.weights[0])
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()

            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost.append(cost)
        return self

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where((np.dot(X, self.weights[1:]) + self.weights[0]) >= 0.0, 1, 0)


#Loading Iris data
X_train, X_test, y_train, y_test = LoadData().load_data()
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionWithGD(eta=0.05, epochs=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
