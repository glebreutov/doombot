import math


import random

import itertools

import operator
import scipy
from numpy.linalg import linalg

import bot_params as params
from PIL import Image
from sklearn import svm
import pickle
import preprocessing as pp
import skimage.feature as ski
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import numpy as np

def save_samples():
    def save_image(data, name):
        img = Image.fromarray(data, 'RGB')
        img.save(params.train_data_folder + "samples/" + name + '.png')

    for i in range(0, 8):
        chosen_one = random.randint(0, memory.inputs.shape[0])
        screen = memory.inputs[chosen_one].reshape(memory.shape)

        prefix = "neg" if memory.outputs[chosen_one] == 0 else "pos"

        save_image(screen, prefix + str(i))


with open(params.train_data_folder + "DepthMemory1469992058_64.pickle", "rb") as fh:
    data = pickle.load(fh)


# for x in data.outputs:
#     for i in range(0, x.size):
#         if x[i] < 201:
#             x[i] += 40
#
#     pp.save_image(x.reshape(240, 320), 'gray_'+str(1), 'L')
#     break
#
# exit()
def count_angle(p1, p2):
    a = abs(p1[0] - p2[0])
    b = abs(p1[1] - p2[1])
    if a == 0 or b == 0:
        return 0
    return int(math.degrees(math.atan(a/b)))

def draw_line(plot, p0, p1):
    dx = abs(p0[0] - p1[0])
    dy = abs(p0[1] - p1[1])
    if dx <= 1 and dy <= 1:
        return
    else:
        plot[p0[1], p0[0]] = 255
        plot[p1[1], p1[0]] = 255
        x = min(p0[0], p1[0]) + dx /2
        y = min(p0[1], p1[1]) + dy /2
        plot[y, x] = 255
        draw_line(plot, (x, y), p0)
        draw_line(plot, (x, y), p1)



def two_lines_is_one(p1, p2, deg):
    return count_angle(p1[0], p2[0]) in (deg -1, deg+1)

def vector_len(x):
    return math.sqrt(x[0]**2+x[1]**2)

def connect_two_lines(l1, l2):
    nl = l1+l2
    return [max(nl), min(nl)]

def connect_all_lines(lst, deg):
    connected_only = list()

    while 0 < len(lst):
        line = list(lst.pop())
        i = 0
        while i<len(lst):
            if two_lines_is_one(line, lst[i], deg):
                line = connect_two_lines(line, list(lst.pop(i)))
            else:
                i += 1
        connected_only.append(line)
    return connected_only

def connect_lines(lines):
    by_angle = dict()
    for line in lines:
        p0, p1 = line
        angle = count_angle(p0, p1)
        if angle not in by_angle:
            by_angle[angle] = list()
        by_angle[angle].append(line)
    conn_lines = list()
    for deg, lst in by_angle.items():

        conn_lines += connect_all_lines(lst, deg)
    return conn_lines


def draw_lines(plot, lines):
    for line in lines:
        p0, p1 = line
        draw_line(plot, p0, p1)
    return plot





def save_samples():
    for i in range(0, len(data.inputs)):
        x = data.inputs[i]
        canny = ski.canny(x.reshape(240, 320), sigma=0.7).astype(np.uint8)*150
        #canny = data.inputs[i].reshape(240, 320)+ canny
        lines = probabilistic_hough_line(canny, threshold=step, line_length=step, line_gap=0)

        lines_by_angle = connect_lines(lines)
        print "orig lines", len(lines)
        print "connected lines", len(lines_by_angle)

        #draw_line(hough, (10, 10), (20, 14))
        pp.save_image(x.reshape(240, 320), 'gray_' + str(i) + '0', 'L')
        pp.save_image(draw_lines(np.zeros(canny.shape, dtype =np.uint8), lines), 'gray_' + str(i) + '1', 'L')
        pp.save_image(draw_lines(np.zeros(canny.shape, dtype =np.uint8), lines_by_angle), 'gray_'+str(i)+'2', 'L')
        #break
        #


def check_grad(val, prev_val):
    new_grad = (val - prev_val)
    if new_grad != 0:
        new_grad /= abs(new_grad)

    return new_grad


def check_gradient(img):
    def check_gradient_by_axis(img, colidx):
        prev_val = -1
        grad = 0
        for i in range(0, img.shape[colidx]):
            if colidx == 0:
                series = img[i:i + 1, :]
            else:
                series = img[:, i:i + 1]

            if series.min() != series.max():
                return False

            if prev_val != -1:
                new_grad = check_grad(series.max(), prev_val)

                if grad == 0:
                    grad = new_grad
                elif grad != new_grad and new_grad != 0:
                    return False

            prev_val = series.max()
        return True

    col_grad = check_gradient_by_axis(img, 0)
    row_grad = check_gradient_by_axis(img, 1)

    return col_grad or row_grad



def img_to_dots(depth_img):
    ys= list()
    xs = list()
    for i in range(0, depth_img.shape[0]):
        for j in range(0, depth_img.shape[1]):
            if depth_img[i, j] != 0:
                ys.append(i)
                xs.append(j)
    return np.asarray(xs), np.asarray(ys)


def draw_edge_svm(img):
    univ = depth_to_edge_univ(img)
    xs, ys = img_to_dots(univ)
    if len(xs) == 0:
        return univ

    clf = svm.SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
        kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    clf.fit(xs.reshape(-2, 1), ys)
    #svm.SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma='auto',
     #   kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

    for y in range(0, univ.shape[1]):
        x = clf.predict(y)[0]
        #x = int(y * w[0]) + w[1]
        if x < univ.shape[1] and y < univ.shape[0] and x >= 0:
            univ[x, y] = 254

    return univ


def draw_edge(img):
    univ = depth_to_edge_univ(img)
    xs, ys = img_to_dots(univ)
    A = np.array([xs, np.ones(xs.size)])
    w = linalg.lstsq(A.T, ys)[0]

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xs, ys)

    for y in range(0, univ.shape[1]):
        x = int(y * w[0]) + w[1]
        if x < univ.shape[1] and y < univ.shape[0]:
            univ[x, y] = 254

    return univ

def draw_edge_dumb(img):
    univ = depth_to_edge_univ(img)
    xs, ys = img_to_dots(univ)
    angles = list()
    for i in range(0, len(xs)):
        x = xs[i]
        y = ys[i]
        for j in range(0, len(xs)):
            xc = xs[j]
            yc = ys[j]
            if not (x == xc and y == yc):
                lx = x + xc
                ly = y + yc
                degrees = math.degrees(math.tanh(lx / ly))

                if degrees != 0:
                    B = 180 - 90 - degrees
                    angles.append(B)

    grouped = dict()
    for g in angles:
        if g in grouped:
            grouped[g] += 1
        else:
            grouped[g] = 1
    sorted_x = sorted(grouped, key=grouped.get, reverse=True)
    angle = sorted_x[0]
    x = xs[0]
    y = ys[0]
    #for x, y in xs, ys:

    for i in range(x, univ.shape[0]-1):
        b = int(y + 1 * math.tanh(math.radians(angle)))
        if b in range(0, univ.shape[0]):
            univ[int(b), i+1] =254

    return univ


def depth_to_edge_univ(img):
    ips = slice(1, None)
    iis = slice(None, -1)
    return depth_to_edge3(img, ips, iis) + depth_to_edge3(img, iis, ips)


def depth_to_edge3(img, ips, iis):
    img_padded_vert = np.zeros(img.shape)
    img_padded_vert[ips, :] = img[iis, :]

    img_padded_hori = np.zeros(img.shape)
    img_padded_hori[:, ips] = img[:, iis]

    vdiv = (img_padded_vert - img)
    hdiv = (img_padded_hori - img)
    div = pow(vdiv * hdiv, 2)
    div[0, :] = 0
    div[:, 0] = 0
    div[-1, :] = 0
    div[:, -1] = 0
    return ((div/div) * 254).astype(np.uint8)



def hough_line(img):
  # Rho and Theta ranges
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos


def highlight_edges_too(img):
    p = np.zeros(img.shape)
    # for i in range(0, img.shape[0]):
    #     row = img[i, :]
    #     pval = 0
    #     delta = np.zeros(img.shape[1] - 1)
    #     for j in range(1, img.shape[1]):
    #         x1 = row[j - 1]
    #         x2 = row[j]
    #         val = int(x1) - int(x2)
    #         delta[j-1] = val
    #
    #     for j in range(1, delta.shape[0] - 1):
    #         if delta[j - 1] != delta[j] and delta[j + 1] != delta[j -1]:
    #             p[i, j] = 254

    for i in range(0, img.shape[1]):
        row = img[:, i]
        pval = 0
        for j in range(1, img.shape[0]):
            x1 = row[j - 1]
            x2 = row[j]
            val = int(x1) - int(x2)
            if j>1 and val != pval:
                p[j, i] = 254
            pval = val

    p = ski.canny(p, sigma=0.7).astype(np.uint8) * 127

    lines = probabilistic_hough_line(p, threshold=0, line_length=0, line_gap=2)
    #p = np.zeros(img.shape)
    for p1, p2 in lines:
        angle = abs(count_angle(p1, p2))
        print angle
        if angle not in(0, 90, 180, 270) and angle < 360:
            draw_line(p, p1, p2)
    #     #if p1[0] != p2[0] and p1[1] != p2[1]:
    return p


def highlight_min_max(img):
    minval = img.min()
    maxval = img.max()
    p = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        row = img[i, :]
        for j in range(1, img.shape[1]):
            x1 = row[j - 1]
            x2 = row[j]
            if abs(int(x1) - int(x2)) == 0:
                p[i, j] = 254

    return p.astype(np.uint8)


def draw_panno(x, y):
    panno = np.zeros(x.shape)
    for i in range(10, 230, step):
        for j in range(10, 310, step):
            xi = x[i:i + step, j:j + step]
            yi = y[i:i + step, j:j + step]
            yi -= yi.min()
            #yi += yi.max()
            #yi *= 10
            #yi +=100
            panno[i:i + step, j:j + step] = highlight_edges_too(yi)
            #if not check_gradient(yi):
                #
                #panno[i:i + step, j:j + step] = draw_edge_svm(yi)
    return panno.astype(np.uint8)


step = 40
for i in range(0, data.inputs.shape[0]):
    x = data.inputs[i].reshape(240, 320)
    y = data.outputs[i].reshape(240, 320)

    #pp.save_image(draw_panno(x, y), 'panno' + str(i) + '0', 'L')
    pp.save_image(highlight_edges_too(y), 'panno' + str(i) + '0', 'L')
    pp.save_image(x, 'panno' + str(i) + '1', 'L')
    if i > 10:
        break

# gray = data.inputs[9].reshape(240, 320)
# depth = data.outputs[9].reshape(240, 320)
#
# pp.save_image(gray, 'gray_ex', 'L')
# pp.save_image(depth, 'depth', 'L')
#
#
