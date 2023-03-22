import sys
sys.path.append('../../')
import numpy as np
from utils.coverages.utils import get_layer_outs_new, percent_str
from collections import defaultdict
import utils.testcase_utils as TestCaseUtils

def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def normalization_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    return X_std




def measure_neuron_cov(model, test_inputs, scaler, threshold=0, skip_layers=None, outs=None):
    if outs is None:
        outs = get_layer_outs_new(self.params,model, test_inputs, skip_layers)
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


class ChangeScorer():
    def __init__(self,params, modelv1,modelv2,test_input,scaler=default_scale,threshold=0.5, skip_layers=None):
        # self.activation_table = defaultdict(float)
        self.params = params
        self.model = modelv1
        self.modelv2 = modelv2
        self.scaler = scaler
        self.threshold = threshold
        # self.activation_table = defaultdict(float)
        self.decay_rate = 0.9
        self.deepgini_value = 1
        self.history_change_score = []
        # self.neuron_change_weight, self.subneuron_change_weight, self.skip_layers = ChangeMeasureUtils.get_neuron_change_awareness(self.model, self.modelv2)
        self.skip_layers = skip_layers#ChangeMeasureUtils.get_skiped_layer(self.model)
        self.seed_activation_table = defaultdict(float)
        self.failure_activation_table = defaultdict(float)
        self.failure_score_table = defaultdict(float)
        # self.activation_behaviour_dict = defaultdict(float)
        self.source_id_banned_dict = defaultdict(float)

    def step(self, test_inputs,update_state=True):
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
        return [self.seed_activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]
        # print('state',state)

    def set_change_score_list(self, score_list):
        self.history_change_score = score_list

    def reset_measure_state(self):
        self.seed_activation_table = defaultdict(float)

    def get_current_score_list(self):
        return self.history_change_score

    def get_category_score(self,given_id):
        # if len(self.history_change_score) == 0:
        #     return 0
        #return np.mean(self.history_change_score)
        return self.seed_activation_table[given_id]
        # return np.mean(self.history_change_score)

    def get_failure_score(self,given_id):
        # if len(self.history_change_score) == 0:
        #     return 0
        #return np.mean(self.history_change_score)
        # print(given_id)
        # print('add hghhhherer')
        # print(self.failure_score_table[given_id])
        return self.failure_score_table[given_id]

        # return np.mean(self.history_change_score)


    def get_current_score(self):
        if len(self.failure_score_table) == 0:
             return 0

        return np.mean(list(self.failure_score_table.values()))

    def get_current_failure_score(self):
        if len(self.failure_score_table) == 0:
            return 0

        return np.mean(list(self.failure_score_table.values()))

    def get_max_score(self):
        if len(self.history_change_score) == 0:
            return 0
        return np.max(self.history_change_score)

    def calculate_change_score(self):
        return 0

    def get_failure_type(self):
        if len(self.failure_activation_table) == 0:
            return 0
        # print(list(self.failure_score_table.keys()))
        return len(self.failure_activation_table)
    # def calculate_score_list(self,test_inputs,score_method='difference'):
    #     outs_m1 = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
    #     outs_m2 = get_layer_outs_new(self.modelv2, test_inputs, self.skip_layers)
    #     layer_nums = len(outs_m1)
    #     if score_method=='mahant':
    #         score = np.sum(abs(outs_m1[-1] - outs_m2[-1]), axis=1)
    #     elif score_method == 'deepgini':
    #         score = (1 - np.sum(outs_m2[-1] ** 2, axis=1)) - (1 - np.sum(outs_m1[-1] ** 2, axis=1))
    #     elif score_method == 'difference':
    #         predict_result_v1 = np.argmax(outs_m1[-1], axis=1)
    #         y_prob_vector_max_confidence_m1 = np.max(outs_m1[-1], axis=1)
    #         y_m2_at_m1_max_pos = []
    #         for i in range(len(predict_result_v1)):
    #             y_m2_at_m1_max_pos.append(outs_m2[-1][i][predict_result_v1[i]])
    #         score = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)
    #     score_list = score.tolist()
    #     return score_list

    def calculate_score_list(self,test_inputs,score_method='difference'):
        outs_m1 = self.model.predict(test_inputs)
        outs_m2 = self.modelv2.predict(test_inputs)
        layer_nums = len(outs_m1)
        if score_method=='mahant':
            score = np.sum(abs(outs_m1 - outs_m2), axis=1)
        elif score_method == 'deepgini':
            score = (1 - np.sum(outs_m2 ** 2, axis=1)) - (1 - np.sum(outs_m1 ** 2, axis=1))
        elif score_method == 'difference':
            predict_result_v1 = np.argmax(outs_m1, axis=1)
            y_prob_vector_max_confidence_m1 = np.max(outs_m1, axis=1)
            y_m2_at_m1_max_pos = []
            for i in range(len(predict_result_v1)):
                y_m2_at_m1_max_pos.append(outs_m2[i][predict_result_v1[i]])
            score = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)
        score_list = score.tolist()
        return score_list

    def initial_seed_list(self, test_case_list):
        test_case_list_len = len(test_case_list)
        test_inputs, test_ids, ground_truths, m1_predicts, m2_predicts, m1_traces, m2_traces = TestCaseUtils.testcaselist2all(test_case_list)
        #test_inputs_preprocessed = ImageUtils.picture_preprocess(test_inputs)
        #score_list = self.calculate_score_list(test_inputs_preprocessed)

        # construct failure dicts
        classnum = len(m2_traces[0])
        delta_traces = m2_traces - m1_traces


        # for i in range(test_case_list_len):
        #     if m1_predicts[i] == ground_truths[i] and m2_predicts[i] != ground_truths[i]:
        #         self.failure_activation_table[(test_ids[i],m1_predicts[i],m2_predicts[i])] = 1
        regression_faults_in_initial_seeds = []
        rest_list = []

        for i in range(test_case_list_len):
            # if there exists regression faults in initial seeds, pick them out
            if m1_predicts[i] == ground_truths[i] and m2_predicts[i] != ground_truths[i]:
                regression_faults_in_initial_seeds.append(test_case_list[i])
                if (test_ids[i], m1_predicts[i], m2_predicts[i]) not in self.failure_activation_table:
                    self.source_id_banned_dict[test_ids[i]] += 1
                    self.failure_activation_table[(test_ids[i],m1_predicts[i],m2_predicts[i])] = 1
            else:
                rest_list.append(test_case_list[i])
            for k in range(classnum):
                if k != m1_predicts[i]:
                    self.failure_score_table[(test_ids[i], m1_predicts[i], k)] = delta_traces[i,k]

        # for i in range(test_case_list_len):
        #     self.activation_table[test_ids[i]] = score_list[i]
        #     if m1_predicts[i] == ground_truths[i] and m2_predicts[i] != ground_truths[i]:
        #         if (test_ids[i],m1_predicts[i],m2_predicts[i]) not in self.activation_behaviour_dict:
        #             self.source_id_banned_dict[test_ids[i]] += 1
        #         self.activation_behaviour_dict[(test_ids[i],m1_predicts[i],m2_predicts[i])] = score_list[i]
        print('initial seeds:',len(test_ids))
        print('initial failures:', len(self.failure_score_table))
        print('initial regression faults:',len(regression_faults_in_initial_seeds))
        # self.history_change_score.extend(score_list)
        # breakpoint()
        return delta_traces,regression_faults_in_initial_seeds, rest_list

    def test(self, test_case_list):
        test_case_list_len = len(test_case_list)
        test_inputs, test_ids, ground_truths, m1_predicts, m2_predicts, m1_traces, m2_traces = TestCaseUtils.testcaselist2all(test_case_list)

        #test_inputs_preprocessed = ImageUtils.picture_preprocess(test_inputs)
        #score_list = self.calculate_score_list(test_inputs_preprocessed)

        # update history score list
        classnum = len(m2_traces[0])
        delta_traces = m2_traces - m1_traces

        for i in range(test_case_list_len):

            if m1_predicts[i] == ground_truths[i] and m2_predicts[i] != ground_truths[i]:
                if (test_ids[i], m1_predicts[i], m2_predicts[i]) not in self.failure_activation_table:
                    # print('new banned direction',(test_ids[i], m1_predicts[i], m2_predicts[i]))
                    self.source_id_banned_dict[test_ids[i]] += 1
                    self.failure_activation_table[(test_ids[i],m1_predicts[i],m2_predicts[i])] = 1

            for k in range(classnum):
                if k != m1_predicts[i]:
                    self.failure_score_table[(test_ids[i], m1_predicts[i], k)] = max(delta_traces[i,k],self.failure_score_table[(test_ids[i], m1_predicts[i], k)])


        # for i in range(test_case_list_len):
        #
        #     self.activation_table[test_ids[i]] = max(self.activation_table[test_ids[i]],score_list[i])
        #     if m1_predicts[i] == ground_truths[i] and m2_predicts[i] != ground_truths[i]:
        #         if (test_ids[i],m1_predicts[i],m2_predicts[i]) not in self.activation_behaviour_dict:
        #             self.source_id_banned_dict[test_ids[i]] += 1
        #             self.activation_behaviour_dict[(test_ids[i],m1_predicts[i],m2_predicts[i])] = score_list[i]
        #         else:
        #             self.activation_behaviour_dict[(test_ids[i], m1_predicts[i], m2_predicts[i])] = max(score_list[i],self.activation_behaviour_dict[(test_ids[i], m1_predicts[i], m2_predicts[i])])
        # # self.history_change_score.extend(score_list)
        # print(self.activation_behaviour_dict)
        return delta_traces

    def test_no_update(self, test_case_list):

        test_inputs, test_ids, ground_truths, m1_predicts, m2_predicts, m1_traces, m2_traces = TestCaseUtils.testcaselist2all(
            test_case_list)
        # test_inputs_preprocessed = ImageUtils.picture_preprocess(test_inputs)
        # score_list = self.calculate_score_list(test_inputs_preprocessed)
        delta_traces = m2_traces - m1_traces
        # score_list = self.calculate_score_list(test_inputs)
        # consider all the neuron
        # once a neuron is activated, the coverage is calculated by the sum of neuron score
        # update the change_awareness_decay_score dictionary
        # outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        # activation_table_of_each_case = [defaultdict() for i in range(len(test_inputs))]
        # activation_table_on_batch = defaultdict()
        #
        # for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        #     test_input_id = 0
        #     for out_for_input in layer_out:  # out_for_input is output of layer for single input
        #         out_for_input = self.scaler(out_for_input)
        #         for neuron_index in range(out_for_input.shape[-1]):
        #             activation_table_of_each_case[test_input_id].setdefault((layer_index, neuron_index),0)
        #             activation_table_on_batch.setdefault((layer_index, neuron_index),0)
        #             #activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 0
        #             #activation_table_on_batch[(layer_index, neuron_index)] = 0
        #             if np.mean(out_for_input[..., neuron_index]) > self.dynamic_threshold[(layer_index, neuron_index)]:
        #                 activation_table_of_each_case[test_input_id][(layer_index, neuron_index)] = 1 #np.mean(out_for_input[..., neuron_index])
        #                 activation_table_on_batch[(layer_index, neuron_index)] += 1
        #         test_input_id += 1
        #
        # test_case_score_list = []
        # for test_case_activation_dict in activation_table_of_each_case:
        #     sum_score = 0
        #     for neuron in test_case_activation_dict:
        #         if test_case_activation_dict[neuron] > 0:
        #             # print('tn:',test_case_activation_dict[neuron])
        #             sum_score += self.activation_table[neuron] * test_case_activation_dict[neuron]
        #     test_case_score_list.append(sum_score)
        # print('nuscore', test_case_score_list)
        return delta_traces

