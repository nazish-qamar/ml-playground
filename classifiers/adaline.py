import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Adaline():
    def __init__(self, eta=0.01, epochs=10, random_state=None, shuffle_flag=True):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.weights = []
        self.weights_initialized = False
        self.shuffle_flag = shuffle_flag
        self.cost = []
        self.random_num_generator = np.random.RandomState(self.random_state)


    def fit_batch(self, X, y):
        self.weights = self.random_num_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.epochs):
            output = self.activation(np.dot(X, self.weights[1:]) + self.weights[0])
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            self.cost.append((errors ** 2).sum() / 2.0)

        return self

    def fit_stochastic(self, X, y):
        self.weights = self.random_num_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.weights_initialized = True
        self.cost = []

        for i in range(self.epochs):
            if self.shuffle_flag:
                X, y = self.shuffle(X, y)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self.update_weights(x_i, target))
            avg_cost = sum(cost) / len(y)
            self.cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.weights_initialized:
            self.weights = self.random_num_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            self.weights_initialized = True

        for i in range(self.epochs):
            cost = []
            if y.ravel().shape[0] > 1:
                for x_i, target in zip(X, y):
                    cost.append(self.update_weights(x_i, target))
            else:
                cost.append(self.update_weights(X, y))

        return self

    def activation(self, X):
        #a linear activation
        return X

    def predict(self, X):
        return np.where(self.activation(np.dot(X, self.weights[1:]) + self.weights[0]) >= 0.0, 1, -1)

    def shuffle(self, X, y):
        """Shuffle training data"""
        r = self.random_num_generator.permutation(len(y))
        return X[r], y[r]

    def update_weights(self, x_i, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(np.dot(x_i, self.weights[1:]) + self.weights[0])
        error = (target - output)
        self.weights[1:] += self.eta * x_i.dot(error)
        self.weights[0] += self.eta * error
        cost = 0.5 * error**2
        return cost


#Load iris data
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 0, -1, 1)    #setosa(-1) and versicolor(1)

# taking only the sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

X_standardized = np.copy(X)
X_standardized[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_standardized[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#Using Batch gradient descent
model = Adaline(eta=0.01, epochs=20, random_state=1).fit_batch(X_standardized, y)
plt.plot(range(1, len(model.cost) + 1), model.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Adaline Model: Batch Gradient Descent')

plt.show()


#Stochastic gradient descent
ada_sgd = Adaline(epochs=15, eta=0.01, random_state=1).fit_stochastic(X_standardized, y)

plt.plot(range(1, len(ada_sgd.cost) + 1), ada_sgd.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Adaline Model: Stochastic Gradient Descent')

plt.show()

#Stochastic gradient descent without reinitializing weights
ada_sgd.partial_fit(X_standardized[0, :], y[0])

plt.plot(range(1, len(ada_sgd.cost) + 1), ada_sgd.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Adaline Model: Stochastic Gradient Descent without Reinitialization')

plt.show()