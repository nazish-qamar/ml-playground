import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LoadData:
    def __init__(self):
        self.sc = StandardScaler()
        self.standardize = None
        self.dataName = ""

    def load_data(self, standardize=False, dataName="iris"):
        self.standardize = standardize
        self.dataName = dataName
        X_train = None
        X_test = None
        y_train = None
        y_test = None

        if self.dataName == "iris":
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

            if self.standardize:
                self.sc.fit(X_train)
                X_train = self.sc.transform(X_train)
                X_test = self.sc.transform(X_test)

            return X_train, X_test, y_train, y_test

        if self.dataName == "wine":
            # Downloading wine data as pandas dataframe
            df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                                  'ml/machine-learning-databases/wine/wine.data',
                                  header=None)
            df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                               'Alcalinity of ash', 'Magnesium', 'Total phenols',
                               'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                               'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                               'Proline']

            print('Class labels', np.unique(df_wine['Class label']))

            X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.3,
                                                                random_state=0,
                                                                stratify=y)

            return df_wine.columns[1:], X_train, X_test, y_train, y_test