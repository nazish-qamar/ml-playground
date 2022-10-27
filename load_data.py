import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class LoadData:
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                          columns=iris['feature_names'] + ['target'])

        # select setosa and versicolor
        y = df.iloc[0:100, 4].values
        y = np.where(y == 0, -1, 1)  # setosa(-1) and versicolor(1)

        # taking only the sepal length and petal length
        X = df.iloc[0:100, [0, 2]].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)
        return X_train, X_test, y_train, y_test
