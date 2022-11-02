from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from load_data import LoadData
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def model_fit(self):
        self.model = DecisionTreeClassifier(criterion='gini',
                                            max_depth=4,
                                            random_state=1)
        self.model.fit(self.X_train, self.y_train)

    def predict_(self):
        print("Predictions: ", self.model.predict(self.X_test))
        print("True Labels: ", self.y_test)

    def tree_visualization(self):
        dot_data = export_graphviz(self.model,
                                   filled=True,
                                   rounded=True,
                                   class_names=['Setosa',
                                                'Versicolor'],
                                   feature_names=['petal length',
                                                  'petal width'],
                                   out_file=None)
        graph = graph_from_dot_data(dot_data)
        graph.write_png('output_files/tree.png')

X_train, X_test, y_train, y_test = LoadData().load_data(standardize=False)
model = DecisionTree(X_train, y_train, X_test, y_test)
model.model_fit()
model.predict_()
model.tree_visualization()