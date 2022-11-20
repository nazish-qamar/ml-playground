from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from helper.load_data import LoadData


class AdaboostModel:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tree = None
        self.ada = None

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

    def train_adaboost_model(self):
        self.tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=1,
                                      random_state=1)

        self.ada = AdaBoostClassifier(base_estimator=self.tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)


    def calculate_accuracy(self):
        tree = self.tree.fit(self.X_train, self.y_train)
        y_train_pred = tree.predict(self.X_train)
        y_test_pred = tree.predict(self.X_test)

        tree_train = accuracy_score(self.y_train, y_train_pred)
        tree_test = accuracy_score(self.y_test, y_test_pred)
        print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

        ada = self.ada.fit(self.X_train, self.y_train)
        y_train_pred = ada.predict(self.X_train)
        y_test_pred = ada.predict(self.X_test)

        ada_train = accuracy_score(self.y_train, y_train_pred)
        ada_test = accuracy_score(self.y_test, y_test_pred)
        print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))


df = LoadData().load_raw_data(dataName="wine")
model = AdaboostModel()
model.preprocess_data(df)
model.train_adaboost_model()
model.calculate_accuracy()