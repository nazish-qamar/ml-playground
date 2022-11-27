import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from helper.load_data import LoadData


def model_plot(model_):
    plt.plot(range(1, model_.n_iter + 1), model_.costs)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()


class LinearRegressionGD:
    def __init__(self, eta=0.001, n_iter=20):
        self.costs = []
        self.eta = eta
        self.n_iter = n_iter
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])

        for i in range(self.n_iter):
            output = np.dot(X, self.weights[1:]) + self.weights[0]
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.costs.append(cost)
        return self

    def predict(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]


df = LoadData().load_raw_data("housing")
X = df[['RM']].values
y = df['MEDV'].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()  # adding new axis as scikit-learn needs 2D array

model_ = LinearRegressionGD()
model_.fit(X_std, y_std)
model_plot(model_)
