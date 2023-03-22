import sys
sys.path.append('../../')
import numpy as np
from utils.coverages.utils import get_layer_outs_new, percent_str
from collections import defaultdict


def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def normalization_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    return X_std




def measure_neuron_cov(params,model, test_inputs, scaler, threshold=0, skip_layers=None, outs=None):
    if outs is None:
        outs = get_layer_outs_new(params,model, test_inputs, skip_layers)
    activation_table = defaultdict(bool)
    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        for out_for_input in layer_out:  # out_for_input is output of layer for single input
            # out_for_input = scaler(out_for_input)

            for neuron_index in range(out_for_input.shape[-1]):
                activation_table[(layer_index, neuron_index)] = activation_table[(layer_index, neuron_index)] or \
                                                                np.mean(out_for_input[..., neuron_index]) > threshold

    covered = len([1 for c in activation_table.values() if c])
    total = len(activation_table.keys())
    return percent_str(covered, total), covered, total, outs


# from change_measure import get_neuron_change_awareness,neuron_dict_distill_by_threshold

class NeuronChangeCoverage():
    def __init__(self,params,modelv1,modelv2,test_input,scaler=default_scale,threshold=0.5, skip_layers=None):
        # self.activation_table = defaultdict(float)
        self.params = params
        self.model = modelv1
        self.modelv2 = modelv2
        self.scaler = scaler
        self.threshold = threshold
        self.activation_table = defaultdict(float)
        self.decay_rate = 0.9
        self.deepgini_value = 1
        self.history_change_score = []


    def step(self, test_inputs, update_state=True):
        if update_state:
            return self.test(test_inputs)
        else:
            return self.test_no_update(test_inputs)


    def calc_reward(self, activation_table):
        activation_values = np.array(list(activation_table.values()))
        #print("activation_values", activation_values)
        covered = np.sum(activation_values > 0)
        reward = np.sum(activation_values)
        # reward = covered
        return reward, covered

    def get_measure_state(self):
        return [self.activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]
        # print('state',state)

    def set_change_score_list(self, score_list):
        self.history_change_score = score_list

    def reset_measure_state(self):
        self.activation_table = defaultdict(float)

    def get_current_score_list(self):
        return self.history_change_score

    def get_current_score(self):
        if len(self.history_change_score) == 0:
            return 0
        return np.mean(self.history_change_score)

    def calculate_change_score(self):
        return 0


    def initial_seed_list(self, test_inputs):
        outs_m1 = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        outs_m2 = get_layer_outs_new(self.modelv2, test_inputs, self.skip_layers)





        activation_table_of_each_case = [defaultdict() for i in range(len(test_inputs))]
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            test_input_id = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                # out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    activation_table_of_each_case[test_input_id].setdefault((layer_index, neuron_index), 0)
                    if np.mean(out_for_input[..., neuron_index]) > self.dynamic_threshold[(layer_index, neuron_index)]:
                        activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 1 # np.mean(out_for_input[..., neuron_index])
                test_input_id += 1
        test_case_score_list = []
        for test_case_activation_dict in activation_table_of_each_case:
            sum_score = 0
            for neuron in test_case_activation_dict:
                if test_case_activation_dict[neuron] > 0:
                    # print('tu:',test_case_activation_dict[neuron])
                    sum_score += self.activation_table[neuron] * test_case_activation_dict[neuron]
            test_case_score_list.append(sum_score)
        # np.save('score_list.npy',np.asarray(test_case_score_list))
        # update average score of history
        self.history_change_score.extend(test_case_score_list)
        return test_case_score_list


    def test(self, test_inputs):
        # consider all the neuron
        # once a neuron is activated, the coverage is calculated by the sum of neuron score
        # update the change_awareness_decay_score dictionary
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        activation_table_of_each_case = [defaultdict() for i in range(len(test_inputs))]
        activation_table_on_batch = defaultdict()

        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            test_input_id = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    # if (layer_index, neuron_index) not in self.activation_table:
                        # self.activation_table[(layer_index, neuron_index)] = 0
                        # activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 0
                        # activation_table_on_batch[(layer_index, neuron_index)] = 0
                    activation_table_of_each_case[test_input_id].setdefault((layer_index, neuron_index),0)
                    activation_table_on_batch.setdefault((layer_index, neuron_index),0)

                    if np.mean(out_for_input[..., neuron_index]) > self.dynamic_threshold[(layer_index, neuron_index)]:
                        activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 1 # np.mean(out_for_input[..., neuron_index]) ## xiugai
                        # activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 1
                        activation_table_on_batch[(layer_index, neuron_index)] += 1
                #         if self.activation_table[(layer_index, neuron_index)] > 0:
                #             pass
                #         else:
                #             self.activation_table[(layer_index, neuron_index)] = self.pagerank_importance[(layer_index, neuron_index)]
                test_input_id += 1

        test_case_score_list = []
        for test_case_activation_dict in activation_table_of_each_case:
            sum_score = 0
            for neuron in test_case_activation_dict:
                if test_case_activation_dict[neuron] > 0:
                    # print('tu:',test_case_activation_dict[neuron])
                    sum_score += self.activation_table[neuron] * test_case_activation_dict[neuron] ####### xiugai
            test_case_score_list.append(sum_score)

        # update neuron weight
        for neuron in activation_table_on_batch:
            if activation_table_on_batch[neuron] > 0:
                self.activation_table[neuron] *= self.decay_rate
        # update average score of history
        self.history_change_score.extend(test_case_score_list)
        # self.average_change_score = (np.mean(test_case_score_list) + self.average_change_score)/2
        return test_case_score_list


    def test_no_update(self, test_inputs):
        # consider all the neuron
        # once a neuron is activated, the coverage is calculated by the sum of neuron score
        # update the change_awareness_decay_score dictionary
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        activation_table_of_each_case = [defaultdict() for i in range(len(test_inputs))]
        activation_table_on_batch = defaultdict()

        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            test_input_id = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    activation_table_of_each_case[test_input_id].setdefault((layer_index, neuron_index),0)
                    activation_table_on_batch.setdefault((layer_index, neuron_index),0)
                    #activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 0
                    #activation_table_on_batch[(layer_index, neuron_index)] = 0
                    if np.mean(out_for_input[..., neuron_index]) > self.dynamic_threshold[(layer_index, neuron_index)]:
                        activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 1 #np.mean(out_for_input[..., neuron_index])
                        activation_table_on_batch[(layer_index, neuron_index)] += 1
                test_input_id += 1

        test_case_score_list = []
        for test_case_activation_dict in activation_table_of_each_case:
            sum_score = 0
            for neuron in test_case_activation_dict:
                if test_case_activation_dict[neuron] > 0:
                    # print('tn:',test_case_activation_dict[neuron])
                    sum_score += self.activation_table[neuron] * test_case_activation_dict[neuron]
            test_case_score_list.append(sum_score)
        # print('nuscore', test_case_score_list)
        return test_case_score_list

