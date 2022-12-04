from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from helper.load_data import LoadData


class RandomForestRegression:
    def __init__(self):
        self.forest = RandomForestRegressor(n_estimators=1000,
                                           criterion='squared_error',
                                           random_state=1,
                                           n_jobs=-1)

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=1)
        self.forest.fit(X_train, y_train)
        y_train_pred = self.forest.predict(X_train)
        y_test_pred = self.forest.predict(X_test)

        print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
        return self


df = LoadData().load_raw_data("housing")
X = df.iloc[:, :-1].values
y = df['MEDV'].values

model_ = RandomForestRegression()
model_.fit(X, y)
