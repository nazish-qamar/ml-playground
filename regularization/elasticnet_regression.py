from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from helper.load_data import LoadData


class ElasticNetRegression:
    def __init__(self):
        self.elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

    def fit(self, X, y):
        self.elanet.fit(X_train, y_train)
        y_train_pred = self.elanet.predict(X_train)
        return self

    def predict(self, X_test):
        y_test_pred = self.elanet.predict(X_test)
        return y_test_pred

df = LoadData().load_raw_data("housing")
X = df[['RM']].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

model_ = ElasticNetRegression()
model_.fit(X_train, y_train)
#print("Actual: ", y_test)
y_predictions = model_.predict(X_test)
#print("Prediction: ", y_predictions)

print('Test MSE : %.3f' % (
        mean_squared_error(y_test, y_predictions)))
print('Test R^2 : %.3f' % (
        r2_score(y_test, y_predictions)))