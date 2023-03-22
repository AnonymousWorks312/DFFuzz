from acgan import acgan_mnist
from acgan import acgan_cifar10


class ACGAN:
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.model = acgan_mnist.ACGAN().discriminator
            self.model.load_weights('acgan/discriminator_epoch100_mnist')
        elif dataset == 'CIFAR10':
            self.model = acgan_cifar10.ACGAN().discriminator
            self.model.load_weights('acgan/discriminator_epoch100_cifar10')
        elif dataset == 'FM':
            self.model = acgan_mnist.ACGAN().discriminator
            self.model.load_weights('acgan/discriminator_epoch100_fm')
        elif dataset == 'SVHN':
            self.model = acgan_cifar10.ACGAN().discriminator
            self.model.load_weights('acgan/discriminator_epoch100_svhn')

    def predict_batch(self, preprocessed_test_inputs):
        # acgan discriminator output format: [true false probability, every number probability]
        result = self.model.predict(preprocessed_test_inputs)[0]

        return result
