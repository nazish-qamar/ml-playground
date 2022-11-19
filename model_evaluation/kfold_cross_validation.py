import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from helper.load_data import LoadData


class KFoldCrossValidation:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def perform_cv(self):
        pipe_lr = make_pipeline(StandardScaler(),
                                LogisticRegression(penalty='l2', random_state=1,
                                                   solver='lbfgs', max_iter=10000))

        kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
        scores = []
        for k, (train, test) in enumerate(kfold):
            pipe_lr.fit(self.X_train[train], self.y_train[train])
            score = pipe_lr.score(self.X_train[test], self.y_train[test])
            scores.append(score)
            print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k + 1, np.bincount(y_train[train]), score))
        print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

        scores = cross_val_score(estimator=pipe_lr,
                                 X=self.X_train,
                                 y=self.y_train,
                                 cv=10,
                                 n_jobs=1)
        print('CV accuracy scores: %s' % scores)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True, dataName="cancer")
cv_ = KFoldCrossValidation(X_train, X_test, y_train, y_test)
cv_.perform_cv()
