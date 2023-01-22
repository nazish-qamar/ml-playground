import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

tf.random.set_seed(1)
np.random.seed(1)


def generate_xor_data():
    x = np.random.uniform(low=-1, high=1, size=(200, 2))
    y = np.ones(len(x))
    y[x[:, 0] * x[:, 1] < 0] = 0

    x_train = x[:100, :]
    y_train = y[:100]
    x_valid = x[100:, :]
    y_valid = y[100:]
    return x_train, y_train, x_valid, y_valid


def plot_performance(model, history, X_val, y_val):
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history['loss'], lw=4)
    plt.plot(history['val_loss'], lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history['binary_accuracy'], lw=4)
    plt.plot(history['val_binary_accuracy'], lw=4)
    plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 3)
    plot_decision_regions(X=X_val, y=y_val.astype(np.int32),
                          clf=model)
    ax.set_xlabel(r'$x_1$', size=15)
    ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(-0.025, 1)
    plt.show()


class XORClassification(tf.keras.Sequential):
    def __init__(self):
        super(XORClassification, self).__init__()

    def add_dense(self, name_ , num_unit_, input_shape_, activation_):
        self.add(tf.keras.layers.Dense(num_unit_, activation=activation_,
                              name=name_, input_shape=input_shape_))


X_train, y_train, X_val, y_val = generate_xor_data()
xor_model = XORClassification()
xor_model.add_dense(name_ = 'L1',
                     num_unit_ = 4,
                     input_shape_ = (2,),
                     activation_ = 'relu')
xor_model.add_dense(name_ = 'L2',
                     num_unit_ = 4,
                     input_shape_ = (),
                     activation_ = 'relu')
xor_model.add_dense(name_ = 'L3',
                     num_unit_ = 4,
                     input_shape_ = (),
                     activation_ = 'relu')
xor_model.add_dense(name_ = 'L4',
                     num_unit_ = 1,
                     input_shape_ = (),
                     activation_ = 'sigmoid')

xor_model.summary()
xor_model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

## train:
hist = xor_model.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=200, batch_size=2, verbose=0)

history = hist.history
plot_performance(xor_model, history, X_val, y_val)