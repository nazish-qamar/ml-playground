import os
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(1)


def load_mnist():
    mnist_bldr = tfds.builder('mnist')
    mnist_bldr.download_and_prepare()
    datasets = mnist_bldr.as_dataset(shuffle_files=False)
    print(datasets.keys())
    return datasets['train'], datasets['test']


def plot_evaluation(hist):
    x_arr = np.arange(len(hist['loss'])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()


class CNN(tf.keras.Sequential):
    def __init__(self):
        super(CNN, self).__init__()

    def add_conv2D(self, name_, filters_, kernel_size_, strides_, padding_, activation_):
        self.add(tf.keras.layers.Conv2D(
            filters=filters_, kernel_size=kernel_size_,
            strides=strides_, padding=padding_,
            name=name_, activation=activation_))

    def add_max_pool(self, name_, size_):
        self.add(tf.keras.layers.MaxPool2D(
            pool_size=size_, name=name_))

    def add_flatten(self):
        self.add(tf.keras.layers.Flatten())

    def add_dense(self, name_, units_, activation_):
        self.add(tf.keras.layers.Dense(
            units=units_, name=name_,
            activation=activation_))

    def add_drop_out(self):
        self.add(tf.keras.layers.Dropout(rate=0.5))

    def add_global_average_pooling_2D(self):
        self.add(tf.keras.layers.GlobalAveragePooling2D())


# Loading and processing data
mnist_train_orig, mnist_test_orig = load_mnist()

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32)))

mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32)))

tf.random.set_seed(1)

mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
                                  reshuffle_each_iteration=False)

mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)

cnn_model = CNN()
cnn_model.add_conv2D(name_='conv_1',
                     filters_=32,
                     kernel_size_=(5, 5),
                     strides_=(1, 1),
                     padding_='same',
                     activation_='relu')
cnn_model.add_max_pool(name_='pool_1',
                       size_=(2, 2))
cnn_model.add_conv2D(name_='conv_2',
                     filters_=64,
                     kernel_size_=(5, 5),
                     strides_=(1, 1),
                     padding_='same',
                     activation_='relu')
cnn_model.add_max_pool(name_='pool_2',
                       size_=(2, 2))
cnn_model.add_flatten()
cnn_model.add_dense(name_='fc_1', units_=1024, activation_='relu')
cnn_model.add_drop_out()
cnn_model.add_dense(name_='fc_2', units_=10, activation_='softmax')

cnn_model.build(input_shape=(None, 28, 28, 1))
cnn_model.compute_output_shape(input_shape=(16, 28, 28, 1))
cnn_model.summary()
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])  # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`

history = cnn_model.fit(mnist_train, epochs=NUM_EPOCHS,
                        validation_data=mnist_valid,
                        shuffle=True)

plot_evaluation(history.history)

test_results = cnn_model.evaluate(mnist_test.batch(20))
print('\nTest Acc. {:.2f}%'.format(test_results[1] * 100))

batch_test = next(iter(mnist_test.batch(12)))
preds = cnn_model(batch_test[0])
preds = tf.argmax(preds, axis=1)
print(preds)

if not os.path.exists('models'):
    os.mkdir('models')

cnn_model.save('models/mnist-cnn.h5')
