import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from helper.load_data import LoadData


class PolynomialRegression:
    def __init__(self):
        self.X_quad = None
        self.X_cubic = None
        self.cubic = None
        self.quadratic = None
        self.regr = LinearRegression()

    def create_polynomial_features(self, X):
        self.quadratic = PolynomialFeatures(degree=2)
        self.cubic = PolynomialFeatures(degree=3)
        self.X_quad = self.quadratic.fit_transform(X)
        self.X_cubic = self.cubic.fit_transform(X)

    def fit_quad(self, X, y):
        X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
        regr = self.regr.fit(self.X_quad, y)
        y_quad_fit = regr.predict(self.quadratic.fit_transform(X_fit))
        quadratic_r2 = r2_score(y, regr.predict(self.X_quad))
        return y_quad_fit, quadratic_r2

    def fit_cubic(self, X, y):
        X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
        regr = self.regr.fit(self.X_cubic, y)
        y_cubic_fit = regr.predict(self.cubic.fit_transform(X_fit))
        cubic_r2 = r2_score(y, regr.predict(self.X_cubic))
        return y_cubic_fit, cubic_r2


df = LoadData().load_raw_data("housing")
X = df[['LSTAT']].values
y = df['MEDV'].values

model_ = PolynomialRegression()
model_.create_polynomial_features(X)
_, R2_quad = model_.fit_quad(X, y)
print("R2 for polynomial regression with degree 2: ", R2_quad)
_, R2_cubic = model_.fit_cubic(X, y)
print("R2 for polynomial regression with degree 2: ", R2_cubic)

