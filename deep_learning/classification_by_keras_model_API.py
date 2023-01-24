import tensorflow as tf
from deep_learning.helpers import generate_xor_data, plot_performance

tf.random.set_seed(1)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        h = self.hidden_1(inputs)
        h = self.hidden_2(h)
        h = self.hidden_3(h)
        return self.output_layer(h)


X_train, y_train, X_val, y_val = generate_xor_data()

model = MyModel()
model.build(input_shape=(None, 2))
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
hist = model.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=200, batch_size=2, verbose=0)

history = hist.history
plot_performance(model, history, X_val, y_val)
