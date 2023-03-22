from gan.gan_mnist import discriminator_model
from gan.gan_cifar10 import discriminator_model_cifar10
import tensorflow as tf
from keras import backend as K
import numpy as np


class GAN:
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.model = discriminator_model()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('gan/discriminator_epoch100_mnist')
        elif dataset == 'CIFAR10':
            self.model = discriminator_model_cifar10()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('gan/discriminator_epoch4_cifar10')
        elif dataset == 'FM':
            self.model = discriminator_model()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('gan/discriminator_epoch100_fm')
        elif dataset == 'SVHN':
            self.model = discriminator_model_cifar10()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('gan/discriminator_epoch16_svhn')

    def predict_batch(self, preprocessed_test_inputs):
        result = self.model.predict(preprocessed_test_inputs)

        return result
