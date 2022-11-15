import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from helper.load_data import LoadData


class LearningCurve:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
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

        self.train_sizes, self.train_scores, self.test_scores = learning_curve(estimator=pipe_lr,
                                                                               X=self.X_train,
                                                                               y=self.y_train,
                                                                               train_sizes=np.linspace(0.1, 1.0, 10),
                                                                               cv=10,
                                                                               n_jobs=1)

        self.train_mean = np.mean(self.train_scores, axis=1)
        self.train_std = np.std(self.train_scores, axis=1)
        self.test_mean = np.mean(self.test_scores, axis=1)
        self.test_std = np.std(self.test_scores, axis=1)

    def curve_plotter(self):
        plt.plot(self.train_sizes, self.train_mean,
                 color='blue', marker='o',
                 markersize=5, label='Training accuracy')

        plt.fill_between(self.train_sizes,
                         self.train_mean + self.train_std,
                         self.train_mean - self.train_std,
                         alpha=0.15, color='blue')

        plt.plot(self.train_sizes, self.test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='Validation accuracy')

        plt.fill_between(self.train_sizes,
                         self.test_mean + self.test_std,
                         self.test_mean - self.test_std,
                         alpha=0.15, color='green')

        plt.grid()
        plt.xlabel('Number of training examples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.ylim([0.8, 1.03])
        plt.tight_layout()
        plt.show()


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True, dataName="cancer")
model_ = LearningCurve(X_train, X_test, y_train, y_test)
model_.trainer()
model_.curve_plotter()
