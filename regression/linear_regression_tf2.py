import numpy as np
import matplotlib.pyplot as plt
from helper.load_data import LoadData
import tensorflow as tf
from sklearn.model_selection import train_test_split


# helper funtions
def mean_squared_error(Y, y_pred):
    return tf.reduce_mean(tf.square(y_pred - Y))


def mean_squared_error_deriv(Y, y_pred):
    return tf.reshape(tf.reduce_mean(2 * (y_pred - Y)), [1, 1])


def h(X, weights, bias):
    return tf.tensordot(X, weights, axes=1) + bias


class LinearRegressionTF2:
    def __init__(self, learning_rate=0.001, num_epochs=3):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_samples = None
        self.batch_size = 10
        self.weights = None
        self.bias = None

    def fit_(self, X, Y):
        self.num_samples = X.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.shuffle(500).repeat(self.num_epochs).batch(self.batch_size)
        iterator = dataset.__iter__()
        num_features = X.shape[1]
        self.weights = tf.random.normal((num_features, 1))
        self.bias = 0

        epochs_plot = list()
        loss_plot = list()

        for i in range(self.num_epochs):
            epoch_loss = list()

            for b in range(int(self.num_samples / self.batch_size)):
                x_batch, y_batch = iterator.get_next()

                output = h(x_batch, self.weights, self.bias)
                epoch_loss.append(mean_squared_error(y_batch, output).numpy())
                dJ_dH = mean_squared_error_deriv(y_batch, output)
                dH_dW = x_batch
                dJ_dW = tf.reduce_mean(dJ_dH * dH_dW)
                dJ_dB = tf.reduce_mean(dJ_dH)

                self.weights -= (self.learning_rate * dJ_dW)
                self.bias -= (self.learning_rate * dJ_dB)

            loss = np.array(epoch_loss).mean()
            epochs_plot.append(i + 1)
            loss_plot.append(loss)

        plt.plot(epochs_plot, loss_plot)
        plt.show()

    def predict_(self, X_test, y_test):
        output_ = h(X_test, self.weights, self.bias)
        labels = y_test

        accuracy_op = tf.metrics.MeanAbsoluteError()
        accuracy_op.update_state(labels, output_)
        print('Mean Absolute Error = {}'.format(accuracy_op.result().numpy()))


df = LoadData().load_raw_data("housing")
continuous_features = df[['RM']].values  # choosing crime rate and average number of rooms as continuous variables

X = np.concatenate([continuous_features], axis=1)
Y = df['MEDV'].values  # median value

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2)
X = tf.constant(train_features, dtype=tf.float32)
Y = tf.constant(train_labels, dtype=tf.float32)

X_test = tf.constant(test_features, dtype=tf.float32)
y_test = tf.constant(test_labels, dtype=tf.float32)

model_ = LinearRegressionTF2()
model_.fit_(X, Y)
model_.predict_(X_test, y_test)