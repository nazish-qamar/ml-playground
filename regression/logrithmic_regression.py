import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from helper.load_data import LoadData


class LogrithmicRegression:
    def __init__(self):
        self.X_log = None
        self.y_sqrt = None
        self.regr = LinearRegression()

    def create_feature(self, X, y):
        self.X_log = np.log(X)
        self.y_sqrt = np.sqrt(y)

    def fit_quad(self, X, y):
        X_fit = np.arange(self.X_log.min() - 1, self.X_log.max() + 1, 1)[:, np.newaxis]
        regr = self.regr.fit(self.X_log, self.y_sqrt)
        y_fit = regr.predict(X_fit)
        log_r2 = r2_score(self.y_sqrt, regr.predict(self.X_log))
        return y_fit, log_r2



df = LoadData().load_raw_data("housing")
X = df[['LSTAT']].values
y = df['MEDV'].values

model_ = LogrithmicRegression()
model_.create_feature(X, y)
_, R2_log = model_.fit_quad(X, y)
print("R2 for logrithmic regression: ", R2_log)

