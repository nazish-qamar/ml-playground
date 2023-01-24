import tensorflow as tf
from deep_learning.helpers import generate_xor_data, plot_performance

tf.random.set_seed(1)


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