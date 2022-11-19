import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from helper.load_data import LoadData


class ROC:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_tuning(self):
        X_train2 = self.X_train[:, [4, 14]]
        pipe_lr = make_pipeline(StandardScaler(),
                                PCA(n_components=2),
                                LogisticRegression(penalty='l2',
                                                   random_state=1,
                                                   solver='lbfgs',
                                                   C=100.0))

        cv = list(StratifiedKFold(n_splits=3).split(self.X_train, self.y_train))

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv):
            probas = pipe_lr.fit(X_train2[train],
                                 self.y_train[train]).predict_proba(X_train2[test])

            fpr, tpr, thresholds = roc_curve(self.y_train[test],
                                             probas[:, 1],
                                             pos_label=1)
            mean_tpr += np.interp(mean_fpr, fpr, tpr) #interpolate ROC
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i + 1, roc_auc))

        plt.plot([0, 1],
                 [0, 1],
                 linestyle='--',
                 color=(0.6, 0.6, 0.6),
                 label='Random guessing')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.plot([0, 0, 1],
                 [0, 1, 1],
                 linestyle=':',
                 color='black',
                 label='Perfect performance')

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.show()


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True, dataName="cancer")
model = ROC(X_train, X_test, y_train, y_test)
model.model_tuning()