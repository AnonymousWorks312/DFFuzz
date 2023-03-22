from dffuzz import DFFuzz
from experiment_builder import get_experiment
import importlib
from utils.utility import merge_object


def load_params(params):
    for params_set in params.params_set:
        m = importlib.import_module("params." + params_set)
        print(m)
        new_params = getattr(m, params_set)
        params = merge_object(params, new_params)
    return params


if __name__ == '__main__':
    import argparse
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser(description="Experiments Script For DeepReFuzz")
    parser.add_argument("--params_set", nargs='*', type=str, default=["cifar10", "vgg16", "change", "dffuzz"],
                        help="see params folder")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "FM", "SVHN"])
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg16", "resnet18", "LeNet5", "Alexnet",
                                                                       "vgg16_adv_cw", "LeNet5_adv_cw","Alexnet_adv_cw","resnet18_adv_cw"])
    # please notice that quant can not be ran by deephunter
    # can not run by deephunter kmnc: svhn_resnet_prune
    parser.add_argument("--coverage", type=str, default="change", choices=["change"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time", type=int, default=1440)
    params = parser.parse_args()

    print(params)
    params = load_params(params)
    params.time_minutes = params.time
    params.time_period = params.time_minutes * 60
    experiment = get_experiment(params)
    experiment.time_list = [i * 240 for i in range(1, params.time // 240 + 1 + 1)]  # need one more?
    print(experiment.time_list)

    if params.framework_name == 'dffuzz':
        dh = DFFuzz(params, experiment)
    else:
        raise Exception("No Framework Provided")

    # initialize directory and paths
    import numpy as np
    import os

    print(os.path.abspath(__file__))
    experiment_dir = str(params.coverage)
    dir_name = 'experiment_' + str(params.framework_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if not os.path.exists(os.path.join(dir_name, experiment_dir)):
        os.mkdir(os.path.join(dir_name, experiment_dir))

    both_fail, regression_faults, weaken = dh.run()
    # print(time.time()-starttime)

    np.save(os.path.join(dir_name, experiment_dir, "bothfail.npy"), np.asarray(both_fail))
    np.save(os.path.join(dir_name, experiment_dir, "regression_faults.npy"), np.asarray(regression_faults))
    np.save(os.path.join(dir_name, experiment_dir, "weaken.npy"), np.asarray(weaken))
    # print(regression_faults[0].input)

    print('TOTAL BOTH:', len(both_fail))
    print('TOTAL REGRESSION:', len(regression_faults))
    print('TOTAL WEAKEN:', len(weaken))
    print('CORPUS', dh.corpus)
    np.save(os.path.join(dir_name, experiment_dir, "corpus.npy"), np.asarray(dh.corpus_list))
    print('ITERATION', dh.experiment.iteration)

    if params.framework_name == 'dffuzz':
        print('SCORE', dh.experiment.coverage.get_failure_type())
    import matplotlib.pyplot as plt

    plt.imshow(regression_faults[0].input)
    plt.show()
