import numpy as np
import matplotlib.pyplot as plt
from helper.load_data import LoadData
import tensorflow as tf
from sklearn.model_selection import train_test_split


class LinearRegressionTF2:
    def __init__(self, eta=0.001, n_iter=20):
        self.costs = []
        self.eta = eta
        self.n_iter = n_iter
        self.weights = None


df = LoadData().load_raw_data("housing")
continuous_features = df[['CRIM','RM', 'TAX']].values #choosing crime rate and average number of rooms as continuous variables
categorical_features = df[ ['CHAS'] ].values #Charles River tract bound or not

X = np.concatenate( [ continuous_features , categorical_features ] , axis=1 )
Y = df['MEDV'].values # median value

train_features , test_features ,train_labels, test_labels = train_test_split( X , Y , test_size=0.2 )
X = tf.constant(train_features, dtype=tf.float32)
Y = tf.constant(train_labels, dtype=tf.float32)

test_X = tf.constant(test_features, dtype=tf.float32)
test_Y = tf.constant(test_labels, dtype=tf.float32)