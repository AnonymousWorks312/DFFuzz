from __future__ import print_function, division

from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np


class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        # 可以通过关键字参数loss_weights或loss来为不同的输出设置不同的损失函数或权值。这两个参数均可为Python的列表或字典。这里我们给loss传递单个损失函数，这个损失函数会被应用于所有输出上
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, batch_size=128):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure inputs
        X_train = (X_train.astype(np.float32)) / 255
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(100):
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / batch_size))

            for index in range(int(X_train.shape[0] / batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                imgs = X_train[index * batch_size:(index + 1) * batch_size]

                # Sample noise as generator input
                noise = np.random.normal(-1, 1, (batch_size, 100))

                # The labels of the digits that the generator tries to create an
                # image representation of
                sampled_labels = np.random.randint(0, 10, (batch_size, 1))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, sampled_labels])

                # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
                img_labels = y_train[index * batch_size:(index + 1) * batch_size]
                fake_labels = 10 * np.ones(img_labels.shape)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

                # If at save interval => save generated image samples
            if (epoch + 1) % 4 == 0:
                self.save_model(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("dcgan_models_mnist/%d.png" % epoch)
        plt.close()

    def save_model(self, epoch):
        self.discriminator.save_weights('acgan_models_mnist/discriminator_epoch' + str(epoch + 1), True)
        self.generator.save_weights('acgan_models_mnist/generator_epoch' + str(epoch + 1), True)
        # def save(model, model_name):
        #     model_path = "saved_model/%s.json" % model_name
        #     weights_path = "saved_model/%s_weights.hdf5" % model_name
        #     options = {"file_arch": model_path,
        #                 "file_weight": weights_path}
        #     json_string = model.to_json()
        #     open(options['file_arch'], 'w').write(json_string)
        #     model.save_weights(options['file_weight'])
        #
        # save(self.generator, "generator")
        # save(self.discriminator, "discriminator")

    def test(self, weight_path):
        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = X_train.astype('float32') / 255.
        print(X_train.shape)

        optimizer = Adam(0.0002, 0.5)
        # 可以通过关键字参数loss_weights或loss来为不同的输出设置不同的损失函数或权值。这两个参数均可为Python的列表或字典。这里我们给loss传递单个损失函数，这个损失函数会被应用于所有输出上
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.discriminator.load_weights(weight_path, skip_mismatch=True)
        print(self.discriminator.predict(X_train[0:2, :, :, None]))


if __name__ == '__main__':
    acgan = ACGAN()
    # acgan.train(batch_size=128)

    acgan.test("discriminator_epoch100")
