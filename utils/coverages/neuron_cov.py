import sys
sys.path.append('../../')
import numpy as np
from utils.coverages.utils import get_layer_outs_new, percent
from collections import defaultdict
import change_measure_utils as ChangeMeasureUtils
def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def normalization_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    return X_std


# def measure_neuron_cov(model, test_inputs, scaler, threshold=0, skip_layers=None, outs=None):
#     if outs is None:
#         outs = get_layer_outs_new(model, test_inputs, skip_layers)
#
#     activation_table = defaultdict(bool)
#
#     for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
#         for out_for_input in layer_out:  # out_for_input is output of layer for single input
#             out_for_input = scaler(out_for_input)
#
#             for neuron_index in range(out_for_input.shape[-1]):
#                 activation_table[(layer_index, neuron_index)] = activation_table[(layer_index, neuron_index)] or\
#                                                                 np.mean(out_for_input[..., neuron_index]) > threshold
#
#     covered = len([1 for c in activation_table.values() if c])
#     total = len(activation_table.keys())
#
#     return percent_str(covered, total), covered, total, outs

from utils.coverages.coverage import AbstractCoverage

class NeuronCoverage(AbstractCoverage):
    def __init__(self, params,model, scaler=default_scale, threshold=0.75, skip_layers=None):
        # self.activation_table = defaultdict(float)
        self.params = params
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        # self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.skip_layers = ChangeMeasureUtils.get_skiped_layer(self.model)

        self.start = 0
        self.bytearray_len = 1
        self.layer_neuron_num = []
        self.layer_start_index = []
        print('skipped layers:',self.skip_layers)
        num = 0
        layer_id = 0
        for layer in self.model.layers:
            if layer_id not in self.skip_layers:
                self.layer_start_index.append(num)
                self.layer_neuron_num.append(int(layer.output.shape[-1]))
                num += int(layer.output.shape[-1] * self.bytearray_len)
            layer_id += 1

        self.total_neuron_num = np.sum(self.layer_neuron_num)
        self.activation_table = np.zeros(self.total_neuron_num, dtype=np.uint8)





    def calc_reward(self, activation_table):
        activation_values = np.array(list(activation_table.values()))
        #print("activation_values", activation_values)
        covered_positions = activation_values == 1
        covered = np.sum(covered_positions)
        reward = covered
        return reward, covered

    def get_measure_state(self):
        return [self.activation_table]
    
    def set_measure_state(self, state):
        self.activation_table = state[0]

    def reset_measure_state(self):
        self.activation_table = defaultdict(float)

    # def get_current_coverage(self, with_implicit_reward=False):
    #     if len(self.activation_table.keys()) == 0:
    #         return 0
    #
    #
    #     reward, covered = self.calc_reward(self.activation_table)
    #     total = len(self.activation_table.keys())
    #     return percent(reward, total)

    # def initial_seed_list(self, test_inputs):
    #     outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
    #     for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
    #         test_input_id = 0
    #         for out_for_input in layer_out:  # out_for_input is output of layer for single input
    #             out_for_input = self.scaler(out_for_input)
    #             for neuron_index in range(out_for_input.shape[-1]):
    #                 if np.mean(out_for_input[..., neuron_index]) > self.threshold:
    #                     self.activation_table[(layer_index, neuron_index)] = 1
    #                 elif (layer_index, neuron_index) not in self.activation_table:
    #                     self.activation_table[(layer_index, neuron_index)] = 0
    #             test_input_id += 1
    #     reward, covered = self.calc_reward(self.activation_table)
    #     total = len(self.activation_table.keys())
    #     return percent_str(reward, total), reward, total, outs

    # def test(self, test_inputs):
    #     outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
    #
    #     for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
    #         for out_for_input in layer_out:  # out_for_input is output of layer for single input
    #             out_for_input = self.scaler(out_for_input)
    #
    #             for neuron_index in range(out_for_input.shape[-1]):
    #                 if self.activation_table[(layer_index, neuron_index)] == 1:
    #                     pass
    #                 elif np.mean(out_for_input[..., neuron_index]) > self.threshold:
    #                     self.activation_table[(layer_index, neuron_index)] = 1
    #
    #     reward, covered = self.calc_reward(self.activation_table)
    #     total = len(self.activation_table.keys())
    #     return percent_str(reward, total), reward, total, outs


    # def test(self, test_inputs):
    #     outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
    #     activation_table_of_each_case = [defaultdict() for i in range(len(test_inputs))]
    #     for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
    #         test_input_id = 0
    #         for out_for_input in layer_out:  # out_for_input is output of layer for single input
    #             out_for_input = self.scaler(out_for_input)
    #             for neuron_index in range(out_for_input.shape[-1]):
    #                 activation_table_of_each_case[test_input_id].setdefault((layer_index, neuron_index), 0)
    #                 if np.mean(out_for_input[..., neuron_index]) > self.threshold:
    #                     self.activation_table[(layer_index, neuron_index)] = 1
    #                     activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 1
    #             test_input_id += 1
    #
    #
    #     reward, covered = self.calc_reward(self.activation_table)
    #     total = len(self.activation_table.keys())
    #     return percent_str(reward, total), reward, total, outs, activation_table_of_each_case


    def test(self, test_inputs):
        ptr = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        # total_size neuron numbers  batch_num :test case num
        outs = get_layer_outs_new(self.params,self.model, test_inputs, self.skip_layers)
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            seed_id = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    if np.mean(out_for_input[..., neuron_index]) > self.threshold:
                        id = self.start + self.layer_start_index[layer_index] + neuron_index * self.bytearray_len + 0
                        ptr[seed_id][id] = 1
                seed_id += 1
        return ptr


    def initial_seed_list(self, test_inputs):
        ptr = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        print(ptr.shape)
        # total_size neuron numbers  batch_num :test case num
        outs = get_layer_outs_new(self.params, self.model, test_inputs, self.skip_layers)
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            seed_id = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    if np.mean(out_for_input[..., neuron_index]) > self.threshold:
                        id = self.start + self.layer_start_index[layer_index] + neuron_index * self.bytearray_len + 0
                        ptr[seed_id][id] = 1
                seed_id += 1
        # np.save('neuron_coverage.npy',ptr)
        ## update activation table
        for ptr_seed in ptr:
            self.activation_table = self.activation_table|ptr_seed
        return ptr

    def get_current_coverage(self):
        covered_positions = self.activation_table > 0
        covered = np.sum(covered_positions)
        return percent(covered, len(self.activation_table))
