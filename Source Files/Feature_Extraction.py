import cv2
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


def ReadImage():
    img = cv2.imread('1.png',  cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


def ConnectedComponentWithColours(line):
    line = cv2.bitwise_not(line)

    se = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
    IdilateText = cv2.dilate(line, se, iterations=4)

    se = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
    Ierode = cv2.erode(IdilateText, se, iterations=4)

    ret, labels = cv2.connectedComponents(Ierode)

    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('dilated.png', IdilateText)
    cv2.imshow('Ieroded.png', Ierode)
    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

    return


def ConnectedComponent(line):
    line = cv2.bitwise_not(line)

    se = cv2.getStructuringElement(cv2.MORPH_DILATE, (7, 7))
    IdilateText = cv2.dilate(line, se, iterations=4)

    se = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
    Ierode = cv2.erode(IdilateText, se, iterations=5)

    # Find connected component again
    comp = cv2.connectedComponentsWithStats(Ierode)
    number_of_labels = comp[0]
    labels = comp[1]
    labelStats = comp[2]
    line = cv2.bitwise_not(line)

    area = [labelStats[i, cv2.CC_STAT_AREA] for i in range(1, number_of_labels)]
    area_avg = np.mean(area)

    left_borders = [labelStats[i, cv2.CC_STAT_LEFT] for i in range(1,len(area)) if area[i] > 20]
    right_borders = [labelStats[i, cv2.CC_STAT_LEFT] + labelStats[i,cv2.CC_STAT_WIDTH] for i in range(1, len(area)) if area[i] > 20]

    width = [labelStats[i, cv2.CC_STAT_WIDTH] for i in range(1, len(area)) if area[i] > 20]
    height = [labelStats[i, cv2.CC_STAT_HEIGHT] for i in range(1, len(area)) if area[i] > 30]
    top = [labelStats[i, cv2.CC_STAT_TOP] for i in range(1, len(area)) if area[i] > 30]

    indices = np.argsort(left_borders)

    distance_ = []
    for i in range(len(indices)-1):
        distance_.append(right_borders[indices[i+1]] - left_borders[indices[i]])

    distance_avg = np.mean(distance_)

    avgwidth = np.average(width)
    medwidth = np.median(width)
    standard_deviation = np.std(width)
    avgheight = np.average(height)
    vincityratio = avgheight/avgwidth

    return distance_avg, avgwidth, avgheight, vincityratio, medwidth


def RatioofBlacktoWhitePixels(line):
    RatioBlacktoWhitePixels = np.sum(line <= 100).astype('int')/np.sum(line > 100).astype('int')
    return RatioBlacktoWhitePixels


def EnclosedRegion(line):

    line = cv2.bitwise_not(line)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = False
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(line)
    Diameter = np.mean([keypoint.size for keypoint in keypoints if keypoint.size > 0])
    Length = math.pi * Diameter
    Area = math.pi * (Diameter**2)/4

    return Area


def Fractal_Features(image, outqueue):
    image = cv2.bitwise_not(image)
    X = []
    Y = []
    for d in range(1, 19):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        Idilate = cv2.dilate(image, se, iterations=1)
        Area = np.count_nonzero(Idilate[:, :] == 255.0)
        Scale = np.log(d)  # x-axis
        Quantity = np.log(Area) - np.log(d)  # y-axis
        X.append(Scale)
        Y.append(Quantity)


    points = []
    X_draw = []
    Y_draw = []

    #Append First Point
    points.append((X[0], Y[0]))
    X_draw.append(X[0])
    Y_draw.append(Y[0])
    del X[0]
    del Y[0]

    #Append Second Point
    points.append((X[0], Y[0]))
    X_draw.append(X[0])
    Y_draw.append(Y[0])
    del X[0]
    del Y[0]

    for end in range(1, 3):  # 3 to get only 3 lines => 4 points
        iterate = len(X)  # how many points left to check upon them
        error = []
        for i in range(0, iterate):  # for each X and point fit line with the beginning point
            a = (Y[i] - Y_draw[end]) / (X[i] - X_draw[end])
            if np.isnan(a):
                continue
            b = Y[i] - (a*X[i])
            # YNew = []
            # for j in range(iterate):  # y=ax+b
            #     Y_cal = (a * X[j]) + b
            #     YNew.append(Y_cal)
            # Vectorised instead of for loop above
            YNew= np.multiply(a,X[0:iterate])+b
            error.append(mean_squared_error(Y, YNew))
        Error_sorted = np.sort(error)
        index = error.index(Error_sorted[0])
        if (iterate - (index + 1)) <= (3 - end):  #0 if last iteration
            if end == 2:
                index = -1
            else:
                index = error.index(Error_sorted[1])
        points.append((X[index], Y[index]))
        X_draw.append(X[index])
        Y_draw.append(Y[index])
        # for re in range(0, index + 1):  # remove points that the line covered
        #     del X[0]
        #     del Y[0]
        # Vectorised code instead of for loop
        del X[0:index+1]
        del Y[0:index+1]


    # Frac_Features = []  # the slope of the 3 lines
    # for i in range(0, 3):
    #     try:
    #         slope = (Y_draw[i + 1] - Y_draw[i]) / (X_draw[i + 1] - X_draw[i])
    #         Frac_Features.append(slope)
    #     except Exception as e:
    #         print(e)
    # Vectorised instead of for loop
    try:
        Frac_Features=  (np.array(Y_draw[1:4]) - np.array(Y_draw[0:3]))/(np.array(X_draw[1:4]) - np.array(X_draw[0:3]))
    except Exception as e:
        print(e)
    # print(Frac_Features[0], Frac_Features[1], Frac_Features[2])
    outqueue.put([Frac_Features[0], Frac_Features[1], Frac_Features[2]])
    return

