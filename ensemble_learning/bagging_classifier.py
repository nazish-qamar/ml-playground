import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from helper.load_data import LoadData


class BaggingModel:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tree = None
        self.bag = None

    def preprocess_data(self, df):
        df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                           'Alcalinity of ash', 'Magnesium', 'Total phenols',
                           'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                           'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                           'Proline']

        df = df[df['Class label'] != 1]

        y = df['Class label'].values
        X = df[['Alcohol', 'OD280/OD315 of diluted wines']].values

        y = LabelEncoder().fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                     test_size=0.2,
                                                                     random_state=1,
                                                                     stratify=y)

    def train_bagging_model(self):
        self.tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=None,
                                      random_state=1)

        self.bag = BaggingClassifier(base_estimator=self.tree,
                                n_estimators=500,
                                max_samples=1.0,
                                max_features=1.0,
                                bootstrap=True,
                                bootstrap_features=False,
                                n_jobs=1,
                                random_state=1)


    def calculate_accuracy(self):
        self.tree = self.tree.fit(self.X_train, self.y_train)
        y_train_pred = self.tree.predict(self.X_train)
        y_test_pred = self.tree.predict(self.X_test)

        tree_train = accuracy_score(self.y_train, y_train_pred)
        tree_test = accuracy_score(self.y_test, y_test_pred)
        print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

        self.bag = self.bag.fit(self.X_train, self.y_train)
        y_train_pred = self.bag.predict(self.X_train)
        y_test_pred = self.bag.predict(self.X_test)

        bag_train = accuracy_score(self.y_train, y_train_pred)
        bag_test = accuracy_score(self.y_test, y_test_pred)
        print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))


df = LoadData().load_raw_data(dataName="wine")
model = BaggingModel()
model.preprocess_data(df)
model.train_bagging_model()
model.calculate_accuracy()