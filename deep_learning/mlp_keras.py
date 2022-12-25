import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot as plt

tf.random.set_seed(1)


def load_iris_data(training_size, batch_size):
    iris, iris_info = tfds.load('iris', with_info=True)
    print(iris_info)

    ds_orig = iris['train']
    ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
    print(next(iter(ds_orig)))

    ds_train_orig = ds_orig.take(100)
    ds_test = ds_orig.skip(100)
    ds_train_orig = ds_train_orig.map(
        lambda x: (x['features'], x['label']))
    ds_test = ds_test.map(
        lambda x: (x['features'], x['label']))

    ds_train = ds_train_orig.shuffle(buffer_size=training_size)
    ds_train = ds_train.repeat()
    ds_train = ds_train.batch(batch_size=batch_size)
    ds_train = ds_train.prefetch(buffer_size=1000)
    return ds_train, ds_test

def plot_training(hist):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(hist['loss'], lw=3)
    ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(hist['accuracy'], lw=3)
    ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.show()

class MLPKeras(tf.keras.Sequential):
    def __init__(self):
        super(MLPKeras, self).__init__()

    def add_dense(self, name_ , num_unit_, input_shape_, activation_):
        self.add(tf.keras.layers.Dense(num_unit_, activation=activation_,
                              name=name_, input_shape=input_shape_))


iris_model = MLPKeras()
iris_model.add_dense(name_ = 'fc1',
                     num_unit_ = 16,
                     input_shape_ = (4,),
                     activation_ = 'sigmoid')
iris_model.add_dense(name_ = 'fc2',
                     num_unit_ = 3,
                     input_shape_ = (),
                     activation_ = 'softmax')
iris_model.summary()
iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])



num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)

ds_train, ds_test = load_iris_data(training_size, batch_size)

history = iris_model.fit(ds_train, epochs=num_epochs,
                         steps_per_epoch=steps_per_epoch,
                         verbose=0)

plot_training(history.history)

results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))

iris_model.save('iris-classifier.h5',
                overwrite=True,
                include_optimizer=True,
                save_format='h5')

#reloading
#iris_model_reloaded = tf.keras.models.load_model('iris-classifier.h5')
