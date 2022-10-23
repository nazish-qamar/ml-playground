import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, eta=0.1, epochs=10, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
          Target values.
        """
        random_nums = np.random.RandomState(self.random_state)
        self.weight = random_nums.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for epoch in range(self.epochs):
            errors = 0
            for x_i, target in zip(X, y):
                delta_weight = self.eta * (target - self.predict(x_i))
                self.weight[1:] += delta_weight * x_i
                self.weight[0] += delta_weight
                errors += int(delta_weight != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 0, -1, 1)    #setosa(-1) and versicolor(1)

# taking only the sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

model = Perceptron(eta=0.1, epochs=10)
model.fit(X, y)

plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.show()