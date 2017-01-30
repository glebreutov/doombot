import numpy as np
import pickle
import bot_params as params
from skimage import color

from PIL import Image
import skimage.feature as ski

def normalize_screen(array):
    array = np.ndarray(100)
    return array.astype(np.float) / 255


def rgbd_to_gray_plus_depth(img):
    grayscale = rgb2gray(img[0:3]).astype(np.uint8)
    depth = img[3]

    return grayscale, depth


def rgb2gray(rgb):
    return rgb[1, ...]*0.21 + rgb[0, ...]*0.72 + rgb[2, ...]*0.07


def divide_by_four_pieces(img):
    shape = img.shape
    return img[:shape[0]/2, :shape[1]/2], \
           np.fliplr(img[:shape[0]/2, shape[1]/2:]), \
           np.flipud(img[shape[0]/2:, :shape[1]/2]), \
           np.flipud(np.fliplr(img[shape[0]/2:, shape[1]/2:]))


def save_image(data, name, itype='RGB'):
    imgd = Image.fromarray(data.astype(np.uint8), itype)
    imgd.save(params.train_data_folder + "samples/" + name + '.png')


def apply_kernel(img, kernel):
    result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(0, img.shape[0] - kernel.shape[0]):
        for j in range(0, img.shape[1] - kernel.shape[1]):
            piece = img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            px = kernel.astype(np.float16) * piece
            # px = np.dot(kernel, piece)
            result[i, j] = int(px.sum())
    return result





'''
iedges = np.asarray([
                     [0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0],
                     ])
edges = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
#kernel2 = np.asarray([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
blur = np.asarray([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
])
zeros = np.zeros((gray.shape[0], 1))
gray_ex = ((np.concatenate((zeros, gray, zeros), axis=1)/48).astype(np.uint8)*48).astype(np.float16)


found_edges = apply_kernel(gray_ex, blur)
#found_edges = apply_kernel(found_edges, iedges)
found_edges = apply_kernel(found_edges, edges)

save_image(gray_ex.astype(np.uint8), 'gray_ex', 'L')
save_image(found_edges, 'edges', 'L')

print edges.shape

#save_image(depth*4, 'depth', 'L')

'''