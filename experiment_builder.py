import numpy as np
import time
import keras
import change_measure_utils as ChangeMeasureUtils


class Experiment:
    pass


def get_experiment(params):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment)
    experiment.model = _get_model(params, experiment)
    experiment.modelv2 = _get_model_v2(params, experiment)
    experiment.coverage = _get_coverage(params, experiment)
    experiment.start_time = time.time()
    experiment.iteration = 0
    experiment.termination_condition = generate_termination_condition(experiment, params)
    experiment.time_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                            210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360]
    # experiment.time_list = [1,2]
    return experiment


def generate_termination_condition(experiment, params):
    start_time = experiment.start_time
    time_period = params.time_period

    def termination_condition():
        c2 = time.time() - start_time > time_period
        return c2

    return termination_condition


def _get_dataset(params, experiment):
    if params.dataset == "MNIST":
        # MNIST DATASET
        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1).astype(np.int16)
        test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
        print('xxxxxxxx', np.max(test_images))
    elif params.dataset == "CIFAR10":
        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
        # train_images = train_images[0:2]
        # test_images = test_images[0:2]
        # train_labels = train_labels[0:2]
        # test_labels = test_labels[0:2]
    elif params.dataset == "FM":
        from keras.datasets import fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1).astype(np.int16)
        test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
    elif params.dataset == "SVHN":
        train_images = np.load('./dataset/svhn_x_train_dc.npy')
        train_labels = np.load('./dataset/svhn_y_train_dc.npy')
        test_images = np.load('./dataset/svhn_x_test_dc.npy')
        test_labels = np.load('./dataset/svhn_y_test_dc.npy')
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
    else:
        raise Exception("Unknown Dataset:" + str(params.dataset))
    return {
        "train_inputs": train_images,
        "train_outputs": train_labels,
        "test_inputs": test_images,
        "test_outputs": test_labels
    }


def _get_model_v2(params, experiment):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if params.model == "LeNet5":
        model = keras.models.load_model(
            'models/mnist_lenet5_0.8tr/train_on_0.2tr/keras_mnist_lenet5_model.014-0.9783.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "LeNet5_adv_cw":
        model = keras.models.load_model('models/mnist_lenet5_advtrain/adv_train/keras_mnist_lenet5_cw_0.9830.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "vgg16":
        model = keras.models.load_model(
            'models/cifar10_vgg16_0.8tr/train_on_0.2tr/keras_cifar10_vgg16_model.017-0.8788.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "vgg16_adv_cw":
        model = keras.models.load_model('models/cifar10_vgg16_advtrain/adv_train/keras_cifar10_vgg16_cw_0.8800.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "resnet18":
        model = keras.models.load_model(
            'models/svhn_resnet18_0.8tr/train_on_0.2tr/keras_svhn_resnet18_model.003-0.9193.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_adv_cw":
        model = keras.models.load_model(
            'models/svhn_resnet18_advtrain/adv_train/keras_svhn_resnet18_cw_model.002-0.9201_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet":
        model = keras.models.load_model('models/fm_alexnet_0.8tr/train_on_0.2tr/keras_fm_alexnet_model.016-0.9034.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_adv_cw":
        model = keras.models.load_model('models/fm_alexnet_advtrain/adv_train/keras_fm_alexnet_cw.002-0.9187_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model


def _get_model(params, experiment):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if params.model == "LeNet5":
        # import tensorflow as tf
        model = keras.models.load_model(
            'models/mnist_lenet5_0.8tr/keras_Fri-May-28-06-01-34-2021.model.008-0.8587.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "LeNet5_adv_cw":
        # import tensorflow as tf
        model = keras.models.load_model(
            'models/mnist_lenet5_advtrain/keras_Mon-Dec-27-09-39-09-2021.model.011-0.9807.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "vgg16":
        # import tensorflow as tf
        model = keras.models.load_model(
            'models/cifar10_vgg16_0.8tr/keras_Sat-Oct-30-01-36-17-2021.model.094-0.8767.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "vgg16_adv_cw":
        # import tensorflow as tf
        model = keras.models.load_model(
            'models/cifar10_vgg16_advtrain/keras_Wed-Mar-24-17-55-44-2021.model.135-0.8792.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
    elif params.model == "resnet18":
        model = keras.models.load_model(
            'models/svhn_resnet18_0.8tr/keras_Tue-Dec-28-20-09-09-2021.model.004-0.8885_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_adv_cw":
        model = keras.models.load_model('models/svhn_resnet18_advtrain/keras_svhn_resnet18_model.006-0.9205_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet":
        model = keras.models.load_model('models/fm_alexnet_0.8tr/keras_fm_alexnet.019-0.8933_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_adv_cw":
        model = keras.models.load_model('models/fm_alexnet_advtrain/keras_fm_alexnet.098-0.9170_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model


def _get_coverage(params, experiment):
    # handle input scaling before giving input to model
    def input_scaler(test_inputs):
        model_lower_bound = params.model_input_scale[0]
        model_upper_bound = params.model_input_scale[1]
        input_lower_bound = params.input_lower_limit
        input_upper_bound = params.input_upper_limit
        scaled_input = (test_inputs - input_lower_bound) / (input_upper_bound - input_lower_bound)
        scaled_input = scaled_input * (model_upper_bound - model_lower_bound) + model_lower_bound
        return scaled_input

    if params.coverage == "change":
        from utils.coverages.change_scorer import ChangeScorer
        # TODO: Skip layers should be determined autoamtically
        import utils.utility as ImageUtils
        test_inputs = ImageUtils.picture_preprocess(experiment.dataset['test_inputs'])
        coverage = ChangeScorer(params, experiment.model, experiment.modelv2, test_inputs, threshold=0.5,
                                skip_layers=ChangeMeasureUtils.get_skiped_layer(experiment.model))  # 0:input, 5:flatten

    else:
        raise Exception("Unknown Coverage" + str(params.coverage))

    # coverage._step = coverage.step
    # coverage.step = lambda test_inputs, *a, **kwa: coverage._step(input_scaler(test_inputs), *a, **kwa)

    return coverage
