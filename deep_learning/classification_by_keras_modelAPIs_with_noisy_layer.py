import tensorflow as tf
from deep_learning.helpers import generate_xor_data, plot_performance

tf.random.set_seed(1)


class NoisyLinearModel(tf.keras.Model):
    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLinearModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hidden_1 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

        self.w = self.add_weight(name='weights',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=False):
        if training:
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            noise = tf.random.normal(shape=(batch, dim),
                                     mean=0.0,
                                     stddev=self.noise_stddev)

            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs
        z = tf.matmul(noisy_inputs, self.w) + self.b
        noisyLayer =  tf.keras.activations.relu(z)
        h = self.hidden_1(noisyLayer)
        h = self.hidden_2(h)
        h = self.hidden_3(h)
        return self.output_layer(h)

    def get_config(self):
        config = super(NoisyLinearModel, self).get_config()
        config.update({'output_dim': self.output_dim,
                       'noise_stddev': self.noise_stddev})
        return config


## testing:
X_train, y_train, X_val, y_val = generate_xor_data()

model = NoisyLinearModel(4, noise_stddev=0.1) #chosen 4 units for Noisy layer
model.build(input_shape=(None, 4))

x = tf.zeros(shape=(1, 2))

model(x, training=True)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
hist = model.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=200, batch_size=2,
                 verbose=0)

plot_performance(model, hist.history, X_val, y_val)
