import traceback
import time
import cv2
import numpy as np
import os
import segment as seg
import Feature_Extraction as fe
from sklearn.neighbors import KNeighborsClassifier
import warnings
from multiprocessing import Process,Queue,freeze_support
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
    out_fractalfeatures_queue = Queue()
    out_midheight_queue = Queue()
    out_transistions_queue = Queue()
    t1 = Process(target=fe.Fractal_Features, args=(image,out_fractalfeatures_queue,))
    t2 = Process(target=seg.get_mid_height, args=(image, out_midheight_queue,))
    t3 = Process(target=seg.get_transitions, args=(image, out_transistions_queue,))
    # slope1, slope2, slope3 = fe.Fractal_Features(image)
    # take, f1, f2, f3, f4, f5, f6 = seg.get_mid_height(image)
    # f7 = seg.get_transitions(image)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    slope1, slope2, slope3=out_fractalfeatures_queue.get()
    f2=out_midheight_queue.get()
    f7=out_transistions_queue.get()
    f8 = f2/f7
    print("I am here")
    # return avg_dist, avgwidth, avgheight, medwidth, area
    return f8, slope1, slope2, slope3


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


def clear_global_lists():
    features.clear()
    class_list.clear()
    feature_list.clear()
    return


def train(index, neigh):
    count_lines = 0
    train_count = 2
    dimensions = 0
    for writer_id in range(1, 4):
        for i in range(train_count):
            try:
                title = 'data/' + str(index) + "/" + str(writer_id) + "/" + str(i + 1) + ".png"
                features, number_of_patterns, dimensions = image_to_features(title)
                feature_list.extend(features)
                class_list.extend(set_writer(writer_id, number_of_patterns))
                count_lines += number_of_patterns
            except Exception as e:
                print('Error occurred in Training title ', title, " in iteration", i, "\n", e)
                traceback.print_exc()
    feature_list_ = np.array(feature_list).reshape((count_lines, dimensions))
    neigh.fit(feature_list_, class_list)
    return


def test(index, neigh):
    try:
        title = 'data/' + str(index) + "/test.png"
        features, number_of_patterns, dimensions = image_to_features(title)
        features = np.array(features).reshape((-1, dimensions))
        predictions = []
        for jk in range(len(features)):
            predictions.append(neigh.predict(features[jk].reshape(1, -1))[0])
        Predictedwriter_id = max(set(predictions), key=predictions.count)
    except Exception as e:
        print('Error occurred in Test :-', e)
    return Predictedwriter_id


def clear_global_lists():
    features.clear()
    class_list.clear()
    feature_list.clear()
    return


def run():
    time_file = open("time-knn.txt", "w")
    Result_file = open("results-knn.txt", "w")
    directory_list = []
    for _, dirnames, _ in os.walk(path_forms):
        directory_list.append(dirnames)
    directory_list = directory_list[0]
    folders = len(directory_list)
    totaltime = 0
    for iteration in range(0, folders):
        clear_global_lists()
        neigh = KNeighborsClassifier(n_neighbors=5)
        index = directory_list[iteration]
        start_time = time.time()
        train(index, neigh)
        result = test(index, neigh)
        diff = round(time.time() - start_time, 2)
        time_file.write(str(diff) + "\n")
        Result_file.write(str(result) + "\n")
        totaltime += diff
    print(totaltime)
    Result_file.close()
    time_file.close()
    return
if __name__ == '__main__':
    freeze_support()
    run()
