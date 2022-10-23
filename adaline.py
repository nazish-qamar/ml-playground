import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Adaline():
    def __init__(self, eta=0.01, epochs=10, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.weights = []
        self.cost = []

    def fit(self, X, y):
        random_nums = np.random.RandomState(self.random_state)
        self.weights = random_nums.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.epochs):
            output = self.activation(np.dot(X, self.weights[1:]) + self.weights[0])
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            self.cost.append((errors ** 2).sum() / 2.0)

        return self


    def activation(self, X):
        #a linear activation
        return X

    def predict(self, X):
        return np.where(self.activation(np.dot(X, self.weights[1:]) + self.weights[0]) >= 0.0, 1, -1)


iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 0, -1, 1)    #setosa(-1) and versicolor(1)

# taking only the sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

model = Adaline(eta=0.0005, epochs=20).fit(X, y)
plt.plot(range(1, len(model.cost) + 1), model.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Adaline Model')

plt.show()