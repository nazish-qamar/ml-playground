import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from helper.load_data import LoadData


class ValidationCurve:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        self.train_sizes = None
        self.train_scores = None
        self.test_scores = None
        self.train_mean = None
        self.train_std = None
        self.test_mean = None
        self.test_std = None

    def trainer(self):
        pipe_lr = make_pipeline(StandardScaler(),
                                LogisticRegression(penalty='l2', random_state=1,
                                                   solver='lbfgs', max_iter=10000))
        self.train_scores, self.test_scores = validation_curve(estimator=pipe_lr,
                                                                                X=self.X_train,
                                                                                y=self.y_train,
                                                                                param_name='logisticregression__C',
                                                                                param_range=self.param_range,
                                                                                cv=10)

        self.train_mean = np.mean(self.train_scores, axis=1)
        self.train_std = np.std(self.train_scores, axis=1)
        self.test_mean = np.mean(self.test_scores, axis=1)
        self.test_std = np.std(self.test_scores, axis=1)

    def curve_plotter(self):
        plt.plot(self.param_range, self.train_mean,
                 color='blue', marker='o',
                 markersize=5, label='Training accuracy')

        plt.fill_between(self.param_range, self.train_mean + self.train_std,
                         self.train_mean - self.train_std, alpha=0.15,
                         color='blue')

        plt.plot(self.param_range, self.test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='Validation accuracy')

        plt.fill_between(self.param_range,
                         self.test_mean + self.test_std,
                         self.test_mean - self.test_std,
                         alpha=0.15, color='green')

        plt.grid()
        plt.xscale('log')
        plt.legend(loc='lower right')
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')
        plt.ylim([0.8, 1.0])
        plt.tight_layout()
        plt.show()


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True, dataName="cancer")
model_ = ValidationCurve(X_train, X_test, y_train, y_test)
model_.trainer()
model_.curve_plotter()
