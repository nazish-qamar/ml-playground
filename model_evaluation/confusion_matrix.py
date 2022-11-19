from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from helper.load_data import LoadData


class ConfusionMatrix:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pipe_svc = None

    def model_trainer(self):
        self.pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
        self.pipe_svc.fit(self.X_train, self.y_train)

    def create_confusion_matrix(self):
        y_pred = self.pipe_svc.predict(self.X_test)
        result = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        print(result)
        print('Precision: %.3f' % precision_score(y_true=self.y_test, y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=self.y_test, y_pred=y_pred))
        print('F1: %.3f' % f1_score(y_true=self.y_test, y_pred=y_pred))


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True, dataName="cancer")
model = ConfusionMatrix(X_train, X_test, y_train, y_test)
model.model_trainer()
model.create_confusion_matrix()