import numpy as np

import gan_util
import time
import gc
import utils.testcase_utils as TestCaseUtils
from mutation_selection_logic import MCMC, Roulette, Random
from dffuzz_mutator import get_mutation_func

now_mutator_names = []
import os
import pickle


class INFO:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        # print('i',i.shape)
        if _i in self.dict:
            # print(self.dict[_i])
            return self.dict[_i]
        else:
            I0, I0_new, state = np.copy(i), np.copy(i), 0
            return I0, I0_new, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]


class TestCase:
    def __init__(self, input, ground_truth, source_id):
        self.input = input
        self.label = ground_truth
        self.source_id = source_id
        self.generation = 0
        self.exploration_multiple = 1
        self.ori_image = input
        self.ref_image = input
        self.m1_trace = []
        self.m2_trace = []
        self.m1_predict_label = -1
        self.m2_predict_label = -1
        self.save_in_queue_time = -1
        self.mutation_trace = []
        self.corpus_id = -1
        self.last_corpus_trace = []
        self.fidelity = -1
        self.select_prob = self.calculate_select_prob(self.generation, 0)

    def update_save_in_queue_time(self, official_saved_time):
        self.save_in_queue_time = official_saved_time

    def set_trace(self, new_m1_trace, new_m2_trace):
        self.m1_trace = new_m1_trace
        self.m2_trace = new_m2_trace
        self.m1_predict_label = np.argmax(self.m1_trace)
        self.m2_predict_label = np.argmax(self.m2_trace)

    def get_test_failure_tend(self):
        # if it is regression faults: return None
        if self.m1_predict_label == self.label and self.m1_predict_label == self.m2_predict_label and self.m1_predict_label != -1 and self.m2_predict_label != -1:
            failure_trace = self.m2_trace - self.m1_trace
            failure_trace[self.m1_predict_label] = -1
            failure_direction = np.argmax(failure_trace)
            return (self.source_id, self.m1_predict_label, failure_direction)
        # else return most likely prone
        else:
            if self.m1_predict_label == self.label and self.m2_predict_label != self.label:
                # print('rf',(self.source_id,self.m1_predict_label,self.m2_predict_label))
                return 'rf'
            return None

    def get_all_test_failure_tends_dicts(self):
        # if it is regression faults: return None
        classnum = len(self.m1_trace)
        failure_trace = self.m2_trace - self.m1_trace
        failure_trace_dicts = {}
        for i in range(classnum):
            if self.m1_predict_label == self.label and self.m2_predict_label != i:
                # failure_trace[self.m1_predict_label] = -1
                # failure_direction = np.argmax(failure_trace)
                failure_trace_dicts[(self.source_id, self.m1_predict_label, i)] = failure_trace[i]
        return failure_trace_dicts

    def get_difference(self):
        epsilon = 1e-7
        return (self.m1_trace[self.label] - self.m2_trace[self.label])  # /(self.m1_trace[self.label]+epsilon)

    def calculate_select_prob(self, generation, select_times, init=1, m=20, finish=0.05):
        epsilon = 1e-7
        delt_times = 10
        if select_times > delt_times:
            return 0  # 0

        alpha = np.log(init / finish) / m
        l = - np.log(init) / alpha
        decay = np.exp(-alpha * (float(generation + select_times / 2.0) + l))

        if decay == np.nan:
            decay = 0  # 0
        return decay


def testcaselist2nparray(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].input))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2sourceid(test_case_list, gran='category'):
    new_input = []
    if gran == 'category':
        for i in range(len(test_case_list)):
            new_input.append(np.asarray(test_case_list[i].source_id))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2labels(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].label))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2generation(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].generation))
    new_input = np.asarray(new_input)
    return new_input


def timing(start_time, duriation):
    MIN = 60
    duriation_sec = duriation * MIN
    now = time.time()
    return now - start_time > duriation_sec


import utils.utility as ImageUtils


class DFFuzz:
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.info = INFO()
        self.last_coverage_state = None
        self.input_shape = params.input_shape
        self.pass_prob = 1
        self.corpus = 0
        self.corpus_list = []
        self.both_failure_case = []
        self.regression_faults = []
        self.weaken_case = []
        self.last_used_mutator = None
        self.Gan = gan_utils.set_gan_mode(self.params.fidelity_mode, self.params.dataset)
        self.mutation_strategy_mode = self.params.mutation_strategy_mode
        self.mutation_strategy = self.set_mutation_strategy_mode()

    # def set_gan_mode(self):
    #     print("***fidelity_mode***", self.params.fidelity_mode)
    #     if self.params.fidelity_mode == "gan":
    #         from gan.GAN_utils import GAN
    #         return GAN(self.params.dataset)
    #     elif self.params.fidelity_mode == "acgan":
    #         return ACGAN(self.params.dataset)
    #     else:
    #         from dcgan.DCGAN_utils import DCGAN
    #         return DCGAN(self.params.dataset)

    def set_mutation_strategy_mode(self):
        print("***mutation_strategy_mode***", self.mutation_strategy_mode)
        if self.mutation_strategy_mode == 'MCMC':
            return MCMC()
        elif self.mutation_strategy_mode == 'Roulette':
            return Roulette()
        else:
            return Random()

    def run(self):
        epsilon = 1e-7
        fidelity_mode = self.params.fidelity_mode
        starttime = time.time()
        I_input = self.experiment.dataset["test_inputs"]
        I_label = self.experiment.dataset["test_outputs"]

        ######### initialize the original corpus
        preprocessed_input = ImageUtils.picture_preprocess(I_input)
        fidelity_list = self.Gan.predict_batch(preprocessed_input)
        fidelity_list = np.squeeze(fidelity_list)

        m1_prob_vector = self.experiment.model.predict(preprocessed_input, batch_size=16)
        if self.params.model == "LeNet5_quant" or self.params.model == "vgg16_quant" or self.params.model == "Alexnet_quant" or self.params.model == "resnet18_quant":
            input_details = self.experiment.modelv2.get_input_details()
            output_details = self.experiment.modelv2.get_output_details()
            input_data = preprocessed_input.astype(np.float32)
            self.experiment.modelv2.resize_tensor_input(input_details[0]['index'],
                                                        [len(preprocessed_input), self.params.input_shape[1],
                                                         self.params.input_shape[2], self.params.input_shape[3]])
            self.experiment.modelv2.allocate_tensors()
            self.experiment.modelv2.set_tensor(input_details[0]['index'], input_data)
            self.experiment.modelv2.invoke()
            output_data = self.experiment.modelv2.get_tensor(output_details[0]['index'])
            m2_prob_vector = np.asarray(output_data)
        else:
            m2_prob_vector = self.experiment.modelv2.predict(preprocessed_input, batch_size=16)

        m1_result = np.argmax(m1_prob_vector, axis=1)
        # m2_result = np.argmax(m2_prob_vector,axis=1)
        I_label = np.squeeze(I_label)
        m1_result = np.squeeze(m1_result)
        good_idx = np.where(I_label == m1_result)  # select correct test cases by m1
        I_list = []
        for i in range(len(I_label)):
            if i in good_idx[0]:
                tc = TestCase(I_input[i], I_label[i], i)
                tc.update_save_in_queue_time(time.time() - starttime)
                # set trace and labels
                tc.set_trace(new_m1_trace=m1_prob_vector[i], new_m2_trace=m2_prob_vector[i])
                tc.fidelity = fidelity_list[i]
                tc.corpus_id = self.corpus
                I_list.append(tc)
                self.corpus_list.append(tc)
                self.corpus += 1
        _, regression_faults_in_initial_seeds, rest_list = self.experiment.coverage.initial_seed_list(I_list)

        # select regression faults in initial seed list
        self.regression_faults.extend(regression_faults_in_initial_seeds)
        for rfis in regression_faults_in_initial_seeds:
            rfis.update_save_in_queue_time(time.time() - starttime)

        T = self.Preprocess(rest_list)
        ### initialization ##
        B, B_id = self.SelectNext(T)
        ####################
        dangerous_source_id = set()
        # B, B_id = self.SelectNext(T)
        # select next test case

        time_list = self.experiment.time_list
        self.experiment.iteration = 0

        while not (self.experiment.termination_condition()):
            if self.experiment.iteration % 100 == 0:
                gc.collect()

            if B_id is None:
                import sys
                print('empty seed pool')
                f = open('txt_1.txt', 'a')
                stop_time = (time.time() - starttime) // 60
                f.writelines('\nAT:' + str(stop_time))
                f.writelines('\nTOTAL BOTH:' + str(len(self.both_failure_case)))
                f.writelines('\nTOTAL REGRESSION:' + str(len(self.regression_faults)))
                f.writelines('\nTOTAL WEAKEN:' + str(len(self.weaken_case)))
                f.writelines('\nITERATION:' + str(self.experiment.iteration))
                f.writelines('\nCORPUS:' + str(self.corpus))
                f.writelines('\nSCORE:' + str(self.experiment.coverage.get_current_score()))
                f.writelines('\nFTYPE:' + str(self.experiment.coverage.get_failure_type()))
                f.close()
                print('AT ' + str(time.time() - starttime) + 'MIN')
                print('TOTAL BOTH', len(self.both_failure_case))
                print('TOTAL REGRESSION', len(self.regression_faults))
                print('TOTAL WEAKEN', len(self.weaken_case))
                print('ITERATION', self.experiment.iteration)
                print('CORPUS', self.corpus)
                print('SCORE:', self.experiment.coverage.get_current_score())
                print('FTYPE:', self.experiment.coverage.get_failure_type())
                experiment_dir = str(self.params.coverage) + '_' + str(self.params.model)
                dir_name = 'experiment_' + str(self.params.framework_name)
                if not os.path.exists(os.path.join(dir_name, experiment_dir)):
                    os.mkdir(os.path.join(dir_name, experiment_dir))
                d = open(os.path.join(dir_name, experiment_dir, 'txt_d2' + str(stop_time) + '.pkl'), 'wb')
                pickle.dump(now_mutator_names, d)
                d.close()
                np.save(os.path.join(dir_name, experiment_dir, str(stop_time) + '_rf'), self.regression_faults)
                np.save(os.path.join(dir_name, experiment_dir, str(stop_time) + '_bf'), self.both_failure_case)
                sys.exit(0)

            self.update_prob(self.experiment.iteration)
            # print("B.shape",B.shape)
            # S = self.Sample(B)
            S = B
            # print('S',S)
            # Ps = self.PowerSchedule(S, self.params.K)
            B_new = []
            # successfully generated cases on this sample
            count_regression = []
            count_both_fail = []
            count_weaken = []
            count_halfweaken = []
            count_fix = []
            if len(time_list) > 0:
                if timing(starttime, time_list[0]):
                    f = open('txt_1.txt', 'a')
                    f.writelines('\nAT:' + str(time_list[0]))
                    f.writelines('\nTOTAL BOTH:' + str(len(self.both_failure_case)))
                    f.writelines('\nTOTAL REGRESSION:' + str(len(self.regression_faults)))
                    f.writelines('\nTOTAL WEAKEN:' + str(len(self.weaken_case)))
                    f.writelines('\nITERATION:' + str(self.experiment.iteration))
                    f.writelines('\nCORPUS:' + str(self.corpus))
                    f.writelines('\nSCORE:' + str(self.experiment.coverage.get_current_score()))
                    f.writelines('\nFTYPE:' + str(self.experiment.coverage.get_failure_type()))
                    f.close()
                    print('AT ' + str(time_list[0]) + ' MIN')
                    print('TOTAL BOTH', len(self.both_failure_case))
                    print('TOTAL REGRESSION', len(self.regression_faults))
                    print('TOTAL WEAKEN', len(self.weaken_case))
                    print('ITERATION', self.experiment.iteration)
                    print('CORPUS', self.corpus)
                    print('SCORE:', self.experiment.coverage.get_current_score())
                    print('FTYPE:', self.experiment.coverage.get_failure_type())

                    experiment_dir = str(self.params.coverage) + '_' + str(self.params.model)
                    dir_name = 'experiment_' + str(self.params.framework_name)
                    if not os.path.exists(os.path.join(dir_name, experiment_dir)):
                        os.mkdir(os.path.join(dir_name, experiment_dir))
                    d = open(os.path.join(dir_name, experiment_dir, 'txt_d2' + str(time_list[0]) + '.pkl'), 'wb')
                    pickle.dump(now_mutator_names, d)
                    d.close()
                    np.save(os.path.join(dir_name, experiment_dir, str(time_list[0]) + '_rf'), self.regression_faults)
                    np.save(os.path.join(dir_name, experiment_dir, str(time_list[0]) + '_bf'), self.both_failure_case)
                    time_list.remove(time_list[0])
                if len(time_list) == 0:
                    import sys
                    sys.exit(0)
            Mutants = []

            for s_i in range(len(S)):
                I = S[s_i]
                if "gan" in fidelity_mode:
                    Mutants.extend(self.mutate_for_GAN_discrimitator(I, fidelity_mode))
                else:
                    for i in range(1, 20 + 1):
                        I_new = self.Mutate_normal(I, fidelity_mode)
                        if I_new is not None and self.isChanged(I, I_new):
                            Mutants.append(I_new)

            if len(Mutants) > 0:
                bflist, rflist, wklist, hwklist, B_new, dangerous_source_id, fixlist = self.isFailedTestList(I, Mutants)
                self.both_failure_case.extend(bflist)
                count_both_fail.extend(bflist)
                self.regression_faults.extend(rflist)
                count_regression.extend(rflist)
                self.weaken_case.extend(wklist)
                count_weaken.extend(wklist)
                count_halfweaken.extend(hwklist)
                count_fix.extend(fixlist)

            # update MCMC mutator score
            if self.mutation_strategy_mode == 'MCMC':
                delt_score = 0.00  # 潜力值变化的相对值的阈值

                k = 1 / 3
                for mt_ in B_new + count_regression:  # 只使用I‘在m1、m2上的距离和I、I’在m2上的距离
                    mt_dif = mt_.get_difference()  # I'm1 - I'm2
                    m2_dif = I.m2_trace[I.label] - mt_.m2_trace[mt_.label]  # Im2 - I'm2
                    # mt_dif = I.m1_trace[I.label] - mt_.m2_trace[mt_.label] # I m1  I' m2
                    delta = (1 - k) * m2_dif + k * mt_dif
                    if delta > 0.0:
                        self.mutation_strategy.mutators[mt_.mutation_trace[-1]].difference_score_total += 1
                        break

            selected_test_case_list = []
            # B_new is the rest test cases which need to be find if it is interesting
            # B_new is (gt->gt)
            if len(B_new) > 0:

                # select strategy
                for i in range(len(B_new)):
                    B_new_failure_dicts = B_new[i].get_all_test_failure_tends_dicts()
                    for failure_id in B_new_failure_dicts:
                        if B_new_failure_dicts[failure_id] > self.experiment.coverage.get_failure_score(failure_id):
                            selected_test_case_list.append(B_new[i])
                            break

            if len(selected_test_case_list) > 0 or len(count_regression) > 0:
                selected_test_case_list_add_rf = []
                selected_test_case_list_add_rf.extend(selected_test_case_list)
                selected_test_case_list_add_rf.extend(count_regression)
                regression_faults_failure_ids, past_mutop, history_corpus_ids_on_rf_branch = TestCaseUtils.testcaselist2pastandfailureid(
                    count_regression)

                self.experiment.coverage.step(selected_test_case_list_add_rf, update_state=True)
                print("upscore:", self.experiment.coverage.get_failure_type())

                # update time of interesting cases
                for ca in selected_test_case_list:
                    ca.update_save_in_queue_time(time.time() - starttime)

                B_c, Bs = T
                max_times_a_seed_selected = 5
                for i in range(len(selected_test_case_list)):
                    # the seed in corpus with higher generation will have less chance to be selected
                    # probability_based_on_mutation_times = min(max_times_a_seed_selected,selected_test_case_list[i].generation)
                    B_c += [0]
                    selected_test_case_list_i = np.asarray(selected_test_case_list)[i:i + 1]
                    Bs += [selected_test_case_list_i]
                    selected_test_case_list[i].corpus_id = self.corpus
                    self.corpus_list.append(selected_test_case_list[i])
                    self.corpus += 1

                # ban source ids
                regression_faults_failure_ids_set = set(regression_faults_failure_ids)
                history_corpus_ids_on_rf_branch_set = set(history_corpus_ids_on_rf_branch)
                # print('regression_faults_failure_ids_set',regression_faults_failure_ids_set)

                if len(regression_faults_failure_ids_set) > 0:
                    print(regression_faults_failure_ids_set)
                    print(history_corpus_ids_on_rf_branch_set)
                    Bs_tc_id = 0
                    for Bs_tc in Bs:
                        Bs_tc_failure = Bs_tc[0].get_test_failure_tend()
                        if (Bs_tc_failure == 'rf' or Bs_tc_failure in regression_faults_failure_ids_set) and Bs_tc[
                            0].corpus_id in history_corpus_ids_on_rf_branch_set:
                            Bs_tc[0].select_prob = 0
                            # ban the failure direction
                        if self.experiment.coverage.source_id_banned_dict[Bs_tc[0].source_id] >= 9:
                            Bs_tc[0].select_prob = 0
                        Bs_tc_id += 1

                    self.BatchPrioritize(T, B_id)

                    #####
                del selected_test_case_list
                del count_regression
                del count_both_fail
                del count_weaken
                del count_halfweaken
                # self.experiment.input_chooser.append(testcaselist2nparray(selected_test_case_list), np.array([-1] * len(B_new)))

            B, B_id = self.SelectNext(T)
            print('iteration:', self.experiment.iteration)
            self.experiment.iteration += 1

        return self.both_failure_case, self.regression_faults, self.weaken_case

    def Preprocess(self, I):
        _I = np.random.permutation(I)
        # print(_I)
        Bs = np.array_split(_I, range(self.params.batch1, len(_I), self.params.batch1))
        # permutation
        # split into batches

        return list(np.zeros(len(Bs))), Bs

    def calc_priority(self, B_ci):
        if B_ci < (1 - self.params.p_min) * self.params.gamma:
            return 1 - B_ci / self.params.gamma
        else:
            return self.params.p_min

    def total_potential(self, S, K):
        deepgini_potential = self.calculate_exploration_potential(S)
        PS = self.PowerSchedule(S, K)
        return np.ceil(PS * deepgini_potential).astype(int).tolist()

    def calculate_exploration_potential(self, S):
        deepgini_value = self.calculate_deepgini(S)
        potentials = 1 / deepgini_value
        return potentials

    def SelectNext(self, T):
        B_c, Bs = T
        # print('B_c:',len(B_c))
        # print('Bs:',len(Bs))
        # B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        B_p = [i[0].select_prob for i in Bs]
        epsilon = 1e-7
        # print('B_p',B_p)
        # c = np.random.choice(len(Bs), p=B_p / (np.sum(B_p)+epsilon))
        if np.sum(B_p) == 0:  # seed pool empty
            return None, None
        c = np.random.choice(len(Bs), p=B_p / (np.sum(B_p)))
        # print('Bs[c]:',(Bs[c]))
        # print('c:',c)
        return Bs[c], c

    def Sample(self, B):
        if len(B) >= self.params.batch2:
            c = np.random.choice(len(B), size=self.params.batch2, replace=False)
            return B[c]
        else:
            return B

    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i].input  # id
            I0, I0_new, state = self.info[I]
            # print('self.info[I]:',state)
            p = self.params.beta * 255 * np.sum(I > 0) - np.sum(np.abs(I - I0_new))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)

        # print('potentials',potentials)
        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))

        return Ps

    def isFailedTest(self, I_new):
        model_v1 = self.experiment.model
        # # print((I_new).shape)
        I_new_input = I_new.input.reshape(-1, 28, 28, 1)

        I_new_input_preprocess = ImageUtils.picture_preprocess(I_new_input)
        temp_result_v1 = model_v1.predict(I_new_input_preprocess)

        predict_result_v1 = np.argmax(temp_result_v1, axis=1)
        y_prob_vector_max_confidence_m1 = np.max(temp_result_v1, axis=1)
        ground_truth = I_new.label

        model_v2 = self.experiment.modelv2
        temp_result_v2 = model_v2.predict(I_new_input_preprocess)
        predict_result_v2 = np.argmax(temp_result_v2, axis=1)
        y_m2_at_m1_max_pos = []
        for i in range(len(temp_result_v2)):
            y_m2_at_m1_max_pos.append(temp_result_v2[i][predict_result_v1[i]])
        difference = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)

        if predict_result_v1 != ground_truth and predict_result_v2 != ground_truth:
            # both wrong
            return 1
        elif predict_result_v1 == ground_truth and predict_result_v2 != ground_truth:
            # regression faults
            return 2
        elif predict_result_v1 == ground_truth and predict_result_v2 == ground_truth and difference > 0.3:
            return 3
        elif predict_result_v1 == ground_truth and predict_result_v2 == ground_truth and difference > 0.15:
            return 4

        else:
            return 0

    def isFailedTestList(self, I, I_new_list):
        model_v1 = self.experiment.model
        # # print((I_new).shape)
        I_new_list_inputs = testcaselist2nparray(I_new_list)
        ground_truth_list = testcaselist2labels(I_new_list)
        I_new_input = I_new_list_inputs.reshape(-1, self.params.input_shape[1], self.params.input_shape[2],
                                                self.params.input_shape[3])
        I_new_input_preprocess = ImageUtils.picture_preprocess(I_new_input)
        # print(I_new_input_preprocess.shape)
        temp_result_v1 = model_v1.predict(I_new_input_preprocess)
        predict_result_v1 = np.argmax(temp_result_v1, axis=1)
        y_prob_vector_max_confidence_m1 = np.max(temp_result_v1, axis=1)
        model_v2 = self.experiment.modelv2
        if self.params.model == "LeNet5_quant" or self.params.model == "vgg16_quant" or self.params.model == "Alexnet_quant" or self.params.model == "resnet18_quant":
            input_details = self.experiment.modelv2.get_input_details()
            output_details = self.experiment.modelv2.get_output_details()
            input_data = I_new_input_preprocess.astype(np.float32)
            self.experiment.modelv2.resize_tensor_input(input_details[0]['index'],
                                                        [len(I_new_input_preprocess), self.params.input_shape[1],
                                                         self.params.input_shape[2], self.params.input_shape[3]])
            self.experiment.modelv2.allocate_tensors()
            self.experiment.modelv2.set_tensor(input_details[0]['index'], input_data)
            self.experiment.modelv2.invoke()
            temp_result_v2 = self.experiment.modelv2.get_tensor(output_details[0]['index'])
        else:
            temp_result_v2 = model_v2.predict(I_new_input_preprocess)
        predict_result_v2 = np.argmax(temp_result_v2, axis=1)
        y_m2_at_m1_max_pos = []
        for i in range(len(temp_result_v2)):
            y_m2_at_m1_max_pos.append(temp_result_v2[i][predict_result_v1[i]])
        difference = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)
        difference_I = np.max(I.m1_trace) - I.m2_trace[I.m1_predict_label]

        both_file_list = []
        regression_faults_list = []
        weaken_faults_list = []
        half_weaken_faults_list = []
        rest_case_list = []
        fix_case_list = []
        potential_source_id = []

        for i in range(len(I_new_list)):
            I_new_list[i].set_trace(new_m1_trace=temp_result_v1[i], new_m2_trace=temp_result_v2[i])
            if predict_result_v1[i] != ground_truth_list[i] and predict_result_v2[i] != ground_truth_list[i]:
                # both wrong
                both_file_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] != ground_truth_list[i]:
                # regression faults
                I_new_list[i].exploration_multiple += 1
                potential_source_id.append(I_new_list[i].source_id)
                regression_faults_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and \
                    difference[i] > 0.3:
                I_new_list[i].exploration_multiple += 1
                rest_case_list.append(I_new_list[i])
                weaken_faults_list.append(I_new_list[i])
                potential_source_id.append(I_new_list[i].source_id)
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and \
                    difference[i] > 0.15:
                I_new_list[i].exploration_multiple += 1
                half_weaken_faults_list.append(I_new_list[i])
                rest_case_list.append(I_new_list[i])
                potential_source_id.append(I_new_list[i].source_id)
            elif (predict_result_v1[i] != ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[
                i]):  # 直接修复fault或m2在ts上增强的更多了

                fix_case_list.append(I_new_list[i])
            elif (predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and
                  ((difference_I > 0 > difference[i] and difference_I - difference[i] > 0.15) or (
                          difference[i] < difference_I < 0 and difference_I - difference[i] > 0.15))):
                # 直接修复fault或m2在ts上增强的更多了
                fix_case_list.append(I_new_list[i])
            else:
                rest_case_list.append(I_new_list[i])
        return both_file_list, regression_faults_list, weaken_faults_list, half_weaken_faults_list, rest_case_list, potential_source_id, fix_case_list

    def isChanged(self, I, I_new):
        return np.any(I.input != I_new.input)

    def isNewImageChanged(self, I, I_new):
        return np.any(I.input != I_new)

    def Predict(self, B_new):
        score_list = self.experiment.coverage.step(B_new, update_state=False)
        return score_list

    def CoverageGain(self, cov):
        return cov > 0

    def BatchPrioritize(self, T, B_id):
        B_c, Bs = T
        B_c[B_id] += 1
        Bs[B_id][0].select_prob = Bs[B_id][0].calculate_select_prob(Bs[B_id][0].generation, B_c[B_id])

    def mutate_for_GAN_discrimitator(self, I, fidelity_mode):
        I_new_list = []
        I_new_last_mutator_list = []
        now_mutator_name = self.mutation_strategy.choose_mutator(self.last_used_mutator)
        now_mutator = get_mutation_func(now_mutator_name)
        self.last_used_mutator = now_mutator_name
        now_mutator_names.append(now_mutator_name)
        for i in range(1, 10 + 1):
            # select mutation operator through MCMC
            I_new = now_mutator(np.copy(I.input)).reshape(*(self.input_shape[1:]))
            I_new = np.clip(I_new, 0, 255)
            I_new_last_mutator_list.append(now_mutator_name)
            I_new_list.append(I_new)

        I_new_fidelity = self.Gan.predict_batch(
            preprocessed_test_inputs=ImageUtils.picture_preprocess(np.asarray(I_new_list)))
        if len(I_new_list) > 1:
            I_new_fidelity = np.squeeze(I_new_fidelity)
        new_case_list = []
        is_mutant_fidelity = False

        for i in range(len(I_new_list)):
            if I_new_fidelity[i] >= self.fidelity_threshold(I, mode=fidelity_mode) and self.isNewImageChanged(
                    I, I_new_list[i]):  # satisfy the fidelity
                new_case = TestCase(I_new_list[i], I.label, I.source_id)
                new_case.generation = I.generation + 1
                new_case.mutation_trace.extend(I.mutation_trace)
                new_case.mutation_trace.append(I_new_last_mutator_list[i])
                new_case.select_prob = new_case.calculate_select_prob(new_case.generation, 0)  # 计算选择ts选择概率
                new_case.fidelity = I.fidelity
                new_case.ori_image = I.ori_image
                new_case.last_corpus_trace.extend(I.last_corpus_trace)
                new_case.last_corpus_trace.append(I.corpus_id)
                new_case_list.append(new_case)
                is_mutant_fidelity = True

        if self.mutation_strategy_mode == 'MCMC':
            self.mutation_strategy.mutators[now_mutator_name].total_select_times += 1
            if is_mutant_fidelity:
                self.mutation_strategy.mutators[now_mutator_name].fidelity_case_num += 1
        elif self.mutation_strategy_mode == "Roulette":
            self.mutation_strategy.mutators[now_mutator_name].total_select_times += 1

        return new_case_list

    def Mutate_normal(self, I, fidelity_mode):
        for i in range(1, self.params.TRY_NUM + 1):
            # select mutation operator through MCMC
            now_mutator_name = self.mutation_strategy.choose_mutator(self.last_used_mutator)
            now_mutator = get_mutation_func(now_mutator_name)
            self.last_used_mutator = now_mutator_name

            I_new = now_mutator(np.copy(I.input)).reshape(*(self.input_shape[1:]))
            I_new = np.clip(I_new, 0, 255)
            I_new_fidelity = self.calculate_fidelity_without_gan(I, I_new, mode=fidelity_mode)

            if I_new_fidelity >= self.fidelity_threshold(I, mode=fidelity_mode):  # satisfy the fidelity
                new_case = TestCase(I_new, I.label, I.source_id)
                new_case.generation = I.generation + 1
                new_case.mutation_trace.extend(I.mutation_trace)
                new_case.mutation_trace.append(now_mutator_name)
                new_case.fidelity = I_new_fidelity
                new_case.ori_image = I.ori_image
                new_case.last_corpus_trace.extend(I.last_corpus_trace)
                new_case.last_corpus_trace.append(I.corpus_id)

                if self.mutation_strategy_mode == 'MCMC':
                    self.mutation_strategy.mutators[now_mutator_name].total_select_times += 1
                    self.mutation_strategy.mutators[now_mutator_name].fidelity_case_num += 1
                elif self.mutation_strategy_mode == "Roulette":
                    self.mutation_strategy.mutators[now_mutator_name].total_select_times += 1

                return new_case
            else:
                if self.mutation_strategy_mode == 'MCMC':
                    self.mutation_strategy.mutators[now_mutator_name].total_select_times += 1
                elif self.mutation_strategy_mode == "Roulette":
                    self.mutation_strategy.mutators[now_mutator_name].total_select_times += 1

        return None

    def calculate_fidelity_without_gan(self, I_old, I_new_input, mode="none"):
        if mode == 'ssim':
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(I_old.ori_image, I_new_input, multichannel=True)
            return ssim
        elif mode == "euclidean":
            euclidean = np.sqrt(np.sum(np.square(np.array(I_old.ori_image) - np.array(I_new_input))))
            return euclidean
        else:
            return 1.0

    def fidelity_threshold(self, I_old, mode='dcgan'):
        if "gan" not in mode:
            return 0.99
        else:
            return I_old.fidelity  # *0.99

    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]

    def f(self, I, I_new, I_origin, state):
        if state == 0:
            l0_ref = (np.sum((I - I_new) != 0))
            linf_ref = np.max(np.abs(I - I_new))
        else:
            l0_ref = (np.sum((I - I_new) != 0)) + (np.sum((I - I_origin) != 0))
            linf_ref = max(np.max(np.abs(I - I_new)), np.max(np.abs(I - I_origin)))
        if (l0_ref < self.params.alpha * np.size(I)):
            return linf_ref <= 255
        else:
            return linf_ref < self.params.beta * 255

    def update_prob(self, iteration):
        if iteration > 5000:
            self.pass_prob = 1
        elif iteration > 10000:
            self.pass_prob = 1
        elif iteration > 20000:
            self.pass_prob = 1
        # 1 1 0.98 0.95
