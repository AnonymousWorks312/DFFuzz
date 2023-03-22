import numpy as np

import utils.utility as ImageUtils

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


def isFailedTestList(dffuzz, I, I_new_list):
    model_v1 = dffuzz.experiment.model
    # # print((I_new).shape)
    I_new_list_inputs = testcaselist2nparray(I_new_list)
    ground_truth_list = testcaselist2labels(I_new_list)
    I_new_input = I_new_list_inputs.reshape(-1, dffuzz.params.input_shape[1], dffuzz.params.input_shape[2],
                                            dffuzz.params.input_shape[3])
    I_new_input_preprocess = ImageUtils.picture_preprocess(I_new_input)
    # print(I_new_input_preprocess.shape)
    temp_result_v1 = model_v1.predict(I_new_input_preprocess)
    predict_result_v1 = np.argmax(temp_result_v1, axis=1)
    y_prob_vector_max_confidence_m1 = np.max(temp_result_v1, axis=1)
    model_v2 = dffuzz.experiment.modelv2
    if dffuzz.params.model == "LeNet5_quant" or dffuzz.params.model == "vgg16_quant" or dffuzz.params.model == "Alexnet_quant" or dffuzz.params.model == "resnet18_quant":
        input_details = dffuzz.experiment.modelv2.get_input_details()
        output_details = dffuzz.experiment.modelv2.get_output_details()
        input_data = I_new_input_preprocess.astype(np.float32)
        dffuzz.experiment.modelv2.resize_tensor_input(input_details[0]['index'],
                                                      [len(I_new_input_preprocess), dffuzz.params.input_shape[1],
                                                       dffuzz.params.input_shape[2], dffuzz.params.input_shape[3]])
        dffuzz.experiment.modelv2.allocate_tensors()
        dffuzz.experiment.modelv2.set_tensor(input_details[0]['index'], input_data)
        dffuzz.experiment.modelv2.invoke()
        temp_result_v2 = dffuzz.experiment.modelv2.get_tensor(output_details[0]['index'])
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
