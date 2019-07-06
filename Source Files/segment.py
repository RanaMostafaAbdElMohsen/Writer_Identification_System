from scipy.stats import norm
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import *
from scipy.spatial import distance
from scipy.signal import *
import logging
from scipy.optimize import curve_fit

path_forms = 'IAM DB/'


def preprocess(image):
    image = cv2.resize(image, (int(np.shape(image)[0] / 2), int(np.shape(image)[1] / 2)))
    image = cv2.GaussianBlur(image, (7, 7), 5)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image


def calculate_horizontal_profiling(image):
    mask = np.uint8(np.where(image == 0, 1, 0))
    row_counts = cv2.reduce(mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
    row_counts = np.transpose(row_counts)
    row_axis = np.arange(np.shape(row_counts)[1],0,-1)
    # make sure they are of the same shape
    row_axis = np.reshape(row_axis,(np.shape(row_counts)))
    return row_counts,row_axis


def calculate_local_maximas_scipy(row_counts,row_axis, order = 20):
    row_counts = np.asarray(row_counts)
    print(row_counts)
    local_maxima = argrelextrema(row_counts, np.greater,order=20)

    final_local = []
    for local in local_maxima[0]:
        count = row_counts[0][local]
        index = row_axis[0][local]
        final_local.append([count,index])

    final_local = np.asarray(final_local)

    return final_local


def calculate_local_maximas(row_counts,row_axis,percent,euclidean_threshold):
    max = np.max(row_counts)
    percent_of_max = percent

    circle_centers = []
    for i in range(np.shape(row_counts)[1]):
        if row_counts[0][i] > percent_of_max * max:
            circle_centers.append([row_counts[0][i], row_axis[0][i]])

    # if the variance in row axis is very small, they belong to the same line
    circle_centers = np.asarray(circle_centers)
    circle_indices = circle_centers[:, 1]
    circle_count = circle_centers[:, 0]

    final_local_maximas = []
    k = 0
    i = 0
    for i in range(len(circle_indices) - 1):
        if distance.euclidean(circle_indices[i], circle_indices[i + 1]) < euclidean_threshold:
            k += 1
        else:
            max_local = np.max(circle_count[i - k:k + i + 1])
            final_local_maximas.append([max_local,circle_indices[i]])
            k = 0

    # the last region
    max = np.max(circle_count[i - k:k + i + 1])
    final_local_maximas.append([max,circle_indices[i]])

    final_local_maximas = np.asarray(final_local_maximas)
    return final_local_maximas


def add_circle_patches(list_circle_centers,ax,image_size):
    for center in list_circle_centers:
        c = Circle((center[0],np.shape(image_size)[0]-center[1]), 5, fill=True, facecolor='None', edgecolor='b', lw=5)
        ax.add_artist(c)

    return


def subplot_image_and_horizontal_profiling(image,row_counts,row_axis,circle_centers):
    plt.subplots(nrows=1, ncols=2, sharex=True)

    plt.subplot(121)
    plt.scatter(row_counts,row_axis)
    plt.xlabel("number of pixels")


    plt.subplot(122)
    plt.imshow(image,cmap='Greys_r',aspect='auto')
    ax = plt.gca()
    add_circle_patches(circle_centers, ax,row_counts)
    plt.show()
    return


def get_image_path(title):
    if title[0] <= 'd':
        sub_directory = "formsA-D/"
    elif title[0] > 'd' and title[0] <= 'h':
        sub_directory = "formsE-H/"
    else:
        sub_directory = "formsI-Z/"

    title = path_forms + sub_directory + title + '.png'

    return title


# red scipy white mask
# blue ours black mask
def compare_scipy_our_script(image):
    # using scipy
    mask_w = np.uint8(np.where(image == 0, 0, 1))
    row_counts = cv2.reduce(mask_w, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
    row_counts = np.asarray(row_counts)
    local_maxima = argrelextrema(row_counts, np.less, order=15)

    # using my script
    count_, axis = calculate_horizontal_profiling(image)
    local_maximas_manual = calculate_local_maximas(count_, axis, 0.1, 20)
    # subplot_image_and_horizontal_profiling(image,count_,axis,local_maximas_manual)

    plt.figure()
    plt.subplot(121)

    row_axis = np.arange(np.shape(row_counts)[0], 0, -1)
    row_counts = np.transpose(row_counts)
    row_axis = np.reshape(row_axis, (np.shape(row_counts)))
    plt.scatter(row_counts, row_axis)
    plt.xlabel("number of pixels")

    plt.subplot(122)
    ax2 = plt.gca()
    for local in local_maxima[0]:
        count = row_counts[0][local] - np.shape(row_counts)[0]
        index = np.shape(row_counts)[1] - row_axis[0][local]
        c = Circle((count, index), 5, fill=True, facecolor='None', edgecolor='r', lw=5)
        ax2.add_artist(c)

    for center in local_maximas_manual:
        c = Circle((center[0], np.shape(image)[0] - center[1]), 5, fill=True, facecolor='None', edgecolor='b', lw=5)
        ax2.add_artist(c)

    plt.imshow(image, aspect="auto", cmap='Greys_r')
    plt.show()

    return local_maximas_manual


def extract_lines(image,local_maxima_coordinates):
    local_maxima_np = np.asarray(local_maxima_coordinates)
    max_pixels = np.max(local_maxima_np[:,1])
    indices = np.shape(image)[0] - local_maxima_np[:,1]
    indices = np.sort(indices)

    start = []
    end = []
    for i in range(len(indices) - 1):
        # some checks for noise removal
        if i > 0:
            if local_maxima_np[i].item(0) >= 0.8 * max_pixels:
                if local_maxima_np[i-1].item(0) >= 0.8 * max_pixels:
                    if local_maxima_np[i + 1].item(0) >= 0.8 * max_pixels:
                        continue
        else:
            if local_maxima_np[i].item(0) >= 0.8 * max_pixels:
                if local_maxima_np[i+1].item(0) >= 0.8 * max_pixels:
                    continue

        start.append(indices[i]+2)
        end.append(indices[i + 1] + 15)

    area = [end[i]-start[i] for i in range(len(start))]
    area_mean = np.mean(area)

    # get component wise variance
    dist = [distance.euclidean(area_mean,area[i]) for i in range(len(start))]
    avg_dist = np.mean(dist)

    lines = []
    for j in range(1,len(start)+1):
        if dist[j-1] > avg_dist * 2.5  and j != 1:
            continue
        im = image[start[j-1]:end[j-1],:]
        lines.append(im)

    return lines


def subplot_lines(lines,image):
    #plt.figure()
    #plt.subplot(121)
    #plt.imshow(image, aspect="auto", cmap='Greys_r')

    for i in range(1,len(lines)+1):
        plt.subplot(len(lines),2,2 * i)
        plt.imshow(lines[i-1],cmap='Greys_r')

    #plt.show()
    return


def segment(image):
    black_count_x,axis_y  = calculate_horizontal_profiling(image)
    # local_maxima = calculate_local_maximas(black_count_x, axis_y, 0.7, 30)
    local_maxima = calculate_local_maximas_scipy(black_count_x,axis_y,10)

    plt.subplots(nrows=1, ncols=2, sharex=True)

    plt.subplot(121)
    plt.scatter(black_count_x, axis_y)
    plt.xlabel("number of pixels")
    ax = plt.gca()
    # add_circle_patches(local_maxima, ax, axis_y)
    for center in local_maxima:
        c = Circle((center[0], center[1]), 5, fill=True, facecolor='None', edgecolor='r', lw=5)
        ax.add_artist(c)

    #plt.subplot(122)
    #plt.imshow(image, cmap='Greys_r', aspect='auto')
    #ax = plt.gca()

    return


def get_peaks(image):
    black_count_x, axis_y = calculate_horizontal_profiling(image)
    black_count_x = np.reshape(black_count_x,(1,np.shape(black_count_x)[1]))
    axis_y = np.reshape(axis_y, (1,np.shape(axis_y)[1]))

    noise_level = 50
    mask = black_count_x <= noise_level
    black_count_x[mask] = 0


    black_count_x = black_count_x.flatten()
    axis_y = axis_y.flatten()
    peaks_indices, prop = find_peaks(black_count_x,height=250,distance=30)
    return black_count_x,axis_y,peaks_indices, prop


def remove_header_footer(black_counts,axis_y,peaks_indices,prop):
    percent = 0.9
    mask = prop['peak_heights'] > percent * np.max(prop['peak_heights'])

    while np.sum(mask.astype(int)) < 3:
        percent -= 0.05
        mask = prop['peak_heights'] > percent * np.max(prop['peak_heights'])

    seperating_lines = peaks_indices[mask]
    start = seperating_lines[1] + 10
    end = seperating_lines[2]

    return start, end


def remove_extra_white_space(black_count,black_axis,start_text,end_text,peak_indices):

    mask_greater = peak_indices > start_text
    mask_less = peak_indices < end_text
    mask = mask_greater & mask_less
    new_peaks = peak_indices[mask]
    new_end = new_peaks[-1] + 50
    return new_end, new_peaks


def change_reference_to_cropped(black_counts,axis_y,peaks,start,end):

    black_count_new = black_counts[start:end]
    axis_y_new = axis_y[start:end]
    peaks_new = peaks-start
    return black_count_new,axis_y_new,peaks_new


def extract_lines_after_crop(black_counts,axis_y,peaks_indices):

    dist = [abs(axis_y[peaks_indices[i]] - axis_y[peaks_indices[i+1]]) for i in range(len(peaks_indices)-1)]
    avgdist = int(np.mean(dist))
    lines_indices = []
    for i in range(len(peaks_indices)):
        start = (np.max(axis_y) - axis_y[peaks_indices[i]]) - int(avgdist/2)
        if start < 0:
            start = 0
        end = start + avgdist
        lines_indices.append([start, end])

    return lines_indices


def subplot_lines_indices(lines,image):

    #plt.figure(4)
    #plt.subplot(121)
    #plt.imshow(image, cmap='Greys_r')

    for i in range(1,len(lines)+1):
        plt.subplot(len(lines),2,2 * i)
        start = lines[i-1][0]
        end = lines[i-1][1]
        im = image[start:end,:]
        plt.imshow(im,cmap='Greys_r')

    #plt.show()
    return




def get_line_images(text_im,lines):
    im = []
    for i in range(len(lines)):
        start = lines[i][0]
        end = lines[i][1]
        im.append(text_im[start:end,:])

    return im


def segment_peaks_finding(image):
    black_count, black_axis, peak_indices, properties = get_peaks(image)
    start_text, end_text = remove_header_footer(black_count, black_axis, peak_indices, properties)
    end_text, text_peaks = remove_extra_white_space(black_count, black_axis, start_text, end_text, peak_indices)
    black_cr, axis_cr, peaks_c = change_reference_to_cropped(black_count, black_axis, text_peaks, start_text, end_text)
    lines = extract_lines_after_crop(black_cr, axis_cr, peaks_c)

    text_im = image[start_text:end_text, :]
    # subplot_lines_indices(lines, text_im)
    line_images = get_line_images(text_im,lines)
    return line_images


def gaussian(x,a,mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def model(x,mu, sigma, height):
    return  height * (1/math.sqrt(2*math.pi)) * (1/sigma) * norm.pdf(x, mu, sigma)

def get_mid_height(line_im,outqueue):

    width = np.shape(line_im)[1]
    im = line_im[:,50:width-50]

    black_counts, axis = calculate_horizontal_profiling(im)
    black_counts = black_counts.flatten()
    axis = axis.flatten()

    noise_level = 5
    mask = black_counts < noise_level
    black_counts[mask] = 0

    lines_draw = []
    top_line = (black_counts > 0).argmax(axis=0)

    mask_2 = black_counts != 0
    bottom_line = black_counts.shape[0] - np.flip(mask_2, axis=0).argmax(axis=0) - 1

    lines_draw.append(top_line)
    lines_draw.append(bottom_line)

    black_counts_normalized = black_counts/np.max(black_counts)

    # mu = np.mean(axis * black_counts)
    mu = np.sum(axis * black_counts_normalized)
    sigma = np.sqrt(np.sum(black_counts_normalized * (axis - mu) ** 2))

    # sigma = np.std(axis)
    # sigma = np.sqrt(np.sum((axis - mu) ** 2))
    popt, pcov = curve_fit(model,axis,black_counts_normalized,p0=[mu,sigma,1])
    gaussian_val = model(axis, *popt)

    take = True
    if popt[1] < 1:
        # print("Sigma fault")
        take = False

    upper_baseline = popt[0]-popt[1]
    lower_baseline = popt[0]+popt[1]
    lines_draw.append(upper_baseline)
    lines_draw.append(lower_baseline)

    f1 = top_line - upper_baseline
    f2 = upper_baseline - lower_baseline
    f3 = lower_baseline - bottom_line

    f4 = f1/f2
    f5 = f1/f3
    f6 = f2/f3
    outqueue.put(f2)
    return



def get_transitions(line_im,outqueue):

    max_len = 0
    max_row = []
    for row in line_im:
        peaks, _ = find_peaks(row)
        if len(peaks) > max_len:
            max_len = len(peaks)
            max_row = row
            max_peaks = peaks

    widths_peaks = peak_widths(max_row,max_peaks)
    f7 = np.median(widths_peaks)
    outqueue.put(f7)
    return

def code_driver_segment():
    filename_forms = "Text Files/writers_forms.txt"
    forms = open(filename_forms,'r')

    for line in forms:
        line = line.split()
        if len(line) == 0:
            continue
        train_count = int(line[2])

        for i in range(train_count):
            title = line[5 + i]
            title = get_image_path(title)
            image = cv2.imread(title, cv2.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = np.asarray(image)

            lines_im = segment_peaks_finding(image)
            for line_im in lines_im:
                # get_mid_height(line_im)
                get_transitions(line_im)
    return
