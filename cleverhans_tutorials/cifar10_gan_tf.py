from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
# import os
# import PIL
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, Flatten, \
    Input, LeakyReLU, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, utils, initializers, backend

# import time
print(tf.__version__)


class Cifar10Gan:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = len(np.unique(y_train))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
            x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
            input_shape = (3, 32, 32)
        else:
            x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
            x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
            input_shape = (32, 32, 3)

        # convert class vectors to binary class matrices
        self.y_train = utils.to_categorical(y_train, num_classes)
        self.y_test = utils.to_categorical(y_test, num_classes)

        # the generator is using tanh activation, for which we need to preprocess
        # the image data into the range between -1 and 1.

        x_train = np.float32(x_train)
        x_train = (x_train / 255 - 0.5) * 2
        self.x_train = np.clip(x_train, -1, 1)

        x_test = np.float32(x_test)
        x_test = (x_test / 255 - 0.5) * 2
        self.x_test = np.clip(x_test, -1, 1)

        # latent space dimension
        self.latent_dim = 100

        self.init = initializers.RandomNormal(stddev=0.02)

        # Generator network
        self.generator = Sequential()
        # Discriminator network
        self.discriminator = Sequential()

        self.epochs = 2
        self.batch_size = 128
        self.smooth = 0.1

        self.real = np.ones(shape=(self.batch_size, 1))
        self.fake = np.zeros(shape=(self.batch_size, 1))

        self.d_loss = []
        self.g_loss = []

        self.z = Input(shape=(self.latent_dim,))
        self.d_g = None

    def build_generator(self):
        # FC: 2x2x512
        self.generator.add(Dense(2 * 2 * 512, input_shape=(self.latent_dim,), kernel_initializer=self.init))
        self.generator.add(Reshape((2, 2, 512)))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(0.2))

        # # Conv 1: 4x4x256
        self.generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(0.2))

        # Conv 2: 8x8x128
        self.generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(0.2))

        # Conv 3: 16x16x64
        self.generator.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(0.2))

        # Conv 4: 32x32x3
        self.generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                                           activation='tanh'))

        # prints a summary representation of your model
        self.generator.summary()

    def build_discriminator(self):
        # imagem shape 32x32x3
        img_shape = self.x_train[0].shape

        # Conv 1: 16x16x64
        self.discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                                      input_shape=img_shape, kernel_initializer=self.init))
        self.discriminator.add(LeakyReLU(0.2))

        # Conv 2:
        self.discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU(0.2))

        # Conv 3:
        self.discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU(0.2))

        # Conv 3:
        self.discriminator.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU(0.2))

        # FC
        self.discriminator.add(Flatten())

        # Output
        self.discriminator.add(Dense(1, activation='sigmoid'))

        # prints a summary representation of your model
        self.discriminator.summary()

        # Optimizer
        self.discriminator.compile(Adam(lr=0.0003, beta_1=0.5), loss='binary_crossentropy',
                                   metrics=['binary_accuracy'])

        self.discriminator.trainable = False

    def loss_function(self):
        img = self.generator(self.z)
        decision = self.discriminator(img)
        self.d_g = Model(inputs=self.z, outputs=decision)
        self.d_g.compile(Adam(lr=0.0004, beta_1=0.5), loss='binary_crossentropy',
                         metrics=['binary_accuracy'])

        # prints a summary representation of your model
        self.d_g.summary()

    def train(self):
        for e in range(self.epochs + 1):
            for i in range(len(self.x_train) // self.batch_size):
                # Train Discriminator weights
                self.discriminator.trainable = True

                # Real samples
                x_batch = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                d_loss_real = self.discriminator.train_on_batch(x=x_batch,
                                                                y=self.real * (1 - self.smooth))

                # Fake Samples
                z = np.random.normal(loc=0, scale=1, size=(self.batch_size, self.latent_dim))
                x_fake = self.generator.predict_on_batch(z)
                d_loss_fake = self.discriminator.train_on_batch(x=x_fake, y=self.fake)

                # Discriminator loss
                d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

                # Train Generator weights
                self.discriminator.trainable = False
                g_loss_batch = self.d_g.train_on_batch(x=z, y=self.real)

                print('epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (
                    e + 1, self.epochs, i, len(self.x_train) // self.batch_size, d_loss_batch, g_loss_batch[0]),
                      100 * ' ', end='\r')

            self.d_loss.append(d_loss_batch)
            self.g_loss.append(g_loss_batch[0])
            print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, self.epochs, self.d_loss[-1], self.g_loss[-1]),
                  100 * ' ')

    def display(self):
        samples = 10
        x_fake = self.generator.predict(np.random.normal(loc=0, scale=1, size=(samples, self.latent_dim)))

        for k in range(samples):
            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
            plt.imshow(((x_fake[k] + 1) * 127).astype(np.uint8))

        plt.tight_layout()
        plt.show()
        # plotting the metrics
        plt.plot(self.d_loss)
        plt.plot(self.g_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Adversarial'], loc='center right')
        plt.show()


def main():
    c1 = Cifar10Gan()
    c1.build_generator()
    c1.build_discriminator()
    c1.loss_function()
    c1.train()
    c1.display()


if __name__ == '__main__':
    main()
