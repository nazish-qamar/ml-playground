import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from helper.load_data import LoadData


def regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return


class DecisionTreeRegression:
    def __init__(self):
        self.tree = DecisionTreeRegressor(max_depth=3)

    def fit(self, X, y):
        self.tree.fit(X, y)
        sort_idx = X.flatten().argsort()
        regplot(X[sort_idx], y[sort_idx], self.tree)
        plt.xlabel('% lower status of the population [LSTAT]')
        plt.ylabel('Price in $1000s [MEDV]')
        plt.show()

        return self


df = LoadData().load_raw_data("housing")
X = df[['LSTAT']].values
y = df['MEDV'].values

model_ = DecisionTreeRegression()
model_.fit(X, y)
