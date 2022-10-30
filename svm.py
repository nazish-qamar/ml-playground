from sklearn.svm import SVC
from load_data import LoadData

X_train_std, X_test_std, y_train, y_test = LoadData().load_data(standardize=True)
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

print("True Class\n",y_test)
print("Predictions\n",svm.predict(X_test_std))