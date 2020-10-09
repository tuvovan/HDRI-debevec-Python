import os
import cv2
import math
import random
import numpy as np


def weight(I):
    I_min, I_max = 0., 255.
    if I <= (I_min + I_max) /2:
        return I - I_min
    return I_max - I

def sample_intensity(stack):
    I_min, I_max = 0., 255.
    num_intensities = int(I_max - I_min + 1)
    num_images = len(stack)
    sample = np.zeros((num_intensities, num_images), dtype=np.uint8)

    mid_img = stack[num_images // 2]

    for i in range(int(I_min), int(I_max + 1)):
        rows, cols  = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(len(stack)):
                sample[i, j] = stack[j][rows[idx], cols[idx]]

    return sample


def estimate_curve(sample, exps, l):
    I_min, I_max = 0., 255.
    n = 255
    A = np.zeros((sample.shape[0] * sample.shape[1] + n, n + sample.shape[0] + 1), dtype=np.float64)
    b = np.zeros((A.shape[0], 1), dtype=np.float64)

    k = 0

    #1. data fitting
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            I_ij = sample[i,j]
            w_ij = weight(I_ij)
            A[k, I_ij] = w_ij
            A[k, n + 1 + i] = -w_ij
            b[k, 0] = w_ij * exps[j]
            k += 1

    #2. smoothing
    for I_k in range(int(I_min + 1), int(I_max)):
        w_k = weight(I_k)
        A[k, I_k-1] = w_k * l
        A[k, I_k] = -2 * w_k * l
        A[k, I_k+1] = w_k * l
        k += 1
    
    #3. Color centering
    A[k, int((I_max - I_min) // 2)] = 1

    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, b)

    g = x[0 : n + 1]

    return g[:,0]


def computeRadiance(stack, exps, curve):
    stack_shape = stack.shape
    img_rad = np.zeros(stack_shape[1:], dtype=np.float64)

    num_imgs = stack_shape[0]

    for i in range(stack_shape[1]):
        for j in range(stack_shape[2]):
            g = np.array([curve[int(stack[k][i, j])] for k in range(num_imgs)])
            w = np.array([weight(stack[k][i, j]) for k in range(num_imgs)])

            sumW = np.sum(w)
            if sumW > 0:
                img_rad[i,j] = np.sum(w * (g - exps) / sumW)
            else:
                img_rad[i,j] = g[num_imgs // 2] - exps[num_imgs //2]
    return img_rad

def globalTonemap(img, l):
    return cv2.pow(img/255., 1.0/l)

def intensityAdjustment(image, template):
    m, n, channel = image.shape
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.average(image[:, :, ch]), np.average(template[:, :, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)

    return output

def load(path_test):
    filenames = []
    exposure_times = []
    f = open(os.path.join(path_test, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        # (filename, exposure, *rest) = line.split()
        (filename, exposure) = line.split()
        filenames += [os.path.join(path_test,filename)]
        # exposure_times += [math.log(float(exposure),2)]
        exposure_times += [float(exposure)]
    return filenames, exposure_times

def read(path_list):
    shape = cv2.imread(path_list[0]).shape

    stack = np.zeros((len(path_list), shape[0], shape[1], shape[2]))
    for i in path_list:
        im = cv2.imread(i)
        stack[path_list.index(i), :, :, :] = im
    return stack