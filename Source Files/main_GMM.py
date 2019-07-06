import time
import cv2
import numpy as np
import math
import segment as seg
import Feature_Extraction as fe
from sklearn.mixture import GaussianMixture
import os
import warnings
warnings.filterwarnings("ignore")
path_forms = 'data/'


def preprocess(image):
    image = cv2.resize(image, (int(np.shape(image)[0] / 2), int(np.shape(image)[1] / 2)))
    image = cv2.GaussianBlur(image, (7, 7), 5)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image


def extract_features(image):
    #avg_dist, avgwidth, avgheight, standard_dev, medwidth= fe.ConnectedComponent(image)
    #area = fe.EnclosedRegion(image)
    slope1, slope2, slope3 = fe.Fractal_Features(image)
    take, f1, f2, f3, f4, f5, f6 = seg.get_mid_height(image)
    f7 = seg.get_transitions(image)
    f8 = f2/f7
    # return avg_dist, avgwidth, avgheight, medwidth, area
    return f1, f2, f4, f5, f6, f8, slope1, slope2


def parse_writers_file_per_line(line):
    line_parsed = line
    Take = True
    test_count = train_count = writer_id = 0
    if len(line_parsed) == 0:
        Take = False
    else:
        test_count = int(line_parsed[4])
        train_count = int(line_parsed[2])
        writer_id = line_parsed[0]
    return Take, writer_id, train_count, test_count


def valid_extracted_features(list_of_feature):
    list_of_feature_arr = np.asarray(list_of_feature)
    if np.shape(list_of_feature_arr) == (1, 1):
        if np.isnan(list_of_feature_arr):
            return False
    else:
        for each_feature in list_of_feature_arr:
            if np.isnan(each_feature):
                return False
    return True


def image_to_features(title_image):
    image = cv2.imread(title_image, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    lines = seg.segment_peaks_finding(image)
    all_feature_list = []
    for sentence in lines:
        feature_per_sentence = extract_features(sentence)
        if valid_extracted_features(feature_per_sentence):
            all_feature_list.extend(feature_per_sentence)
    number_patterns = len(all_feature_list) / len(feature_per_sentence)
    dimensionality = len(feature_per_sentence)
    # all_feature_list = np.reshape(all_feature_list,(1,number_patterns * dimensionality))
    return all_feature_list, int(number_patterns), int(dimensionality)


def set_writer(writerid, number_patterns):
    writer_list = []
    for i in range(int(number_patterns)):
        writer_list.append(writerid)
    return writer_list


features = []
class_list = []
feature_list = []
gaussian_models = []
gaussian_class = []


def train(index):
    count_lines = 0
    train_count = 2
    dimensions = 0
    for writer_id in range(1, 4):
        gaussian_features = []
        for i in range(train_count):
            try:
                title = 'data/' + str(index) + "/" + str(writer_id) + "/" + str(i+1) + ".png"
                features, number_of_patterns, dimensions = image_to_features(title)
                #print(features)
                feature_list.extend(features)
                class_list.extend(set_writer(writer_id, number_of_patterns))
                count_lines += number_of_patterns
                g_features = np.reshape(features, (number_of_patterns, dimensions))
                gaussian_features.extend(g_features)
            except Exception as e:
                print('Error occurred in Training title ', title, " in iteration", i, "\n", e)

        gauss = GaussianMixture(n_components=1)
        gaussian_features = np.reshape(gaussian_features, (-1, dimensions))
        gaussian_models.append(gauss.fit(gaussian_features))
        gaussian_class.append(writer_id)
    return


def test(index):
    try:
        title = 'data/' + str(index) + "/test.png"
        features, number_of_patterns, dimensions = image_to_features(title)
        features = np.array(features).reshape((number_of_patterns, dimensions))
        predictions = []
        confidences = []
        for p in range(number_of_patterns):
            scores = []
            for gauss in gaussian_models:
                g_feat = features[p].reshape(1, -1)
                score = gauss.score(g_feat)
                scores.append(score)

            sorted_indices = np.argsort(scores)
            max_index = sorted_indices[-1]
            second_best = sorted_indices[-2]

            # get covariance of each gaussian component of gauss of max index
            components = 1
            variance_gaussian_component = []
            for c in range(0, components):
                variances = [gaussian_models[max_index].covariances_[c][s, s] for s in range(dimensions)]
                variances_avg = np.mean(variances)
                weighted_var = variances_avg * gaussian_models[max_index].weights_[c]
                variance_gaussian_component.append(weighted_var)
            #std_max_index = np.sqrt(np.sum(variance_gaussian_component))
            confidence_measure_sigma = (math.log(abs(scores[max_index]), 10) - math.log(abs(scores[second_best]), 10))
            writer_predicted = gaussian_class[max_index]
            if abs(confidence_measure_sigma) < 0.5:
                confidences.append(0)
                continue
            else:
                confidences.append(1)
            predictions.append(writer_predicted)
        sum_confidence = np.sum(confidences)
        if sum_confidence == 0:
            Predictedwriter_id = -1
        else:
            Predictedwriter_id = max(set(predictions), key=predictions.count)

    except Exception as e:
        print('Error occurred in Test:-', e)

    return Predictedwriter_id


def clear_global_lists():
    features.clear()
    class_list.clear()
    feature_list.clear()
    gaussian_models.clear()
    gaussian_class.clear()
    return


def run():
    time_file = open("time-gmm.txt", "w")
    Result_file = open("results-gmm.txt", "w")
    directory_list = []
    for _, dirnames, _ in os.walk(path_forms):
        directory_list.append(dirnames)
    directory_list = directory_list[0]
    folders = len(directory_list)
    totaltime = 0
    for iteration in range(0, folders):
        clear_global_lists()
        index = directory_list[iteration]
        start_time = time.time()
        train(index)
        result = test(index)
        diff = round(time.time() - start_time, 2)
        time_file.write(str(diff) + "\n")
        Result_file.write(str(result) + "\n")
        totaltime += diff
    print(totaltime)
    Result_file.close()
    time_file.close()
    return

run()
