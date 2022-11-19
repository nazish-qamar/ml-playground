import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from helper.load_data import LoadData


class GridSearchHyperParamTuning:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # for C parameter

    def fit_with_best_params(self):
        param_grid = [{'svc__C': self.param_range,
                       'svc__kernel': ['linear']},
                      {'svc__C': self.param_range,
                       'svc__gamma': self.param_range,
                       'svc__kernel': ['rbf']}]

        pipe_svc = make_pipeline(StandardScaler(),
                                 SVC(random_state=1))

        gs = GridSearchCV(estimator=pipe_svc,
                          param_grid=param_grid,
                          scoring='accuracy',
                          refit=True,
                          cv=10,
                          n_jobs=-1)
        gs = gs.fit(X_train, y_train)
        print(gs.best_score_)
        print(gs.best_params_)
        return gs

    def test_set_accuracy(self, clf):
        print('Test accuracy: %.3f' % clf.score(self.X_test, self.y_test))


X_train, X_test, y_train, y_test = LoadData().load_data(standardize=True, dataName="cancer")
grid_obj = GridSearchHyperParamTuning(X_train, X_test, y_train, y_test)
gs_ = grid_obj.fit_with_best_params()
clf = gs_.best_estimator_
grid_obj.test_set_accuracy(clf)

#to perfrom nested cross validation we can use cross validation on grid search object
scores = cross_val_score(gs_, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))