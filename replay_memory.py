import pickle
from time import time

import numpy as np
from neon.data import ArrayIterator
import random
import bot_params as params
import preprocessing as ppr
import os


class GenericMemory:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def prepare_input(self, input_data):
        return input_data.reshape((1, -1))

    def prepare_output(self, output):
        return np.asarray([output]).reshape((1, -1))

    def add_episode(self, input_data, output):
        if self.inputs is None:
            self.inputs = self.prepare_input(input_data)
            self.outputs = self.prepare_output(output)
        else:
            self.inputs = np.append(self.inputs, self.prepare_input(input_data), axis=0)
            self.outputs = np.append(self.outputs, self.prepare_output(output), axis=0)

        if self.inputs.shape[0] >= params.num_examples_to_dump:
            self.save_data()
            self.__init__()

    def save_data(self, suffix  =""):
        filename = params.train_data_folder + self.__class__.__name__ + str(long(time())) + suffix + ".pickle"
        print filename
        with open(filename, 'wb') as f:
            pickle.dump(self, f)





class ReplayMemory(GenericMemory):
    def __init__(self, nclass):
        GenericMemory.__init__(self)
        self.shape = None
        self.nclass = nclass

    def add_episode(self, screen, is_killed):
        self.shape = screen.shape

        screen = screen.reshape(1, -1)

        super.add_episode(screen, is_killed)

        if is_killed > 0:
            print "Positive example found. Examples collected:", self.inputs.shape[0]

    def _cook_data(self):
        self.inputs = self.inputs.astype(np.float16) / 255
        orig_shape = self.inputs.shape
        shape4d = self.inputs.shape[0:1] + self.shape
        transpose_dim = (0, 3, 1, 2)
        self.inputs = self.inputs.reshape(shape4d).transpose(transpose_dim)
        transposed_shape = self.inputs.shape
        self.inputs = self.inputs.reshape(orig_shape)
        self.shape = transposed_shape[1:]

        # self.outputs = self.outputs.astype(np.float16)

        tmp_array =[]
        for x in self.outputs:
            tmp_array.append([x == 1])

        self.outputs = np.asarray(tmp_array)

    def _split_dataset(self):
        test_inputs = None
        test_outputs = None

        desired_examples = self.inputs.shape[0]/5

        while True:
            random_idx = random.randrange(0, self.inputs.shape[0])

            chosen_input = self.inputs[random_idx:random_idx+1]

            if test_inputs is None:
                test_inputs = chosen_input
                test_outputs = self.outputs[random_idx:random_idx+1]
            else:
                test_inputs = np.append(test_inputs, chosen_input, axis=0)
                test_outputs = np.append(test_outputs, self.outputs[random_idx:random_idx+1])

            self.inputs = np.delete(self.inputs, random_idx, axis=0)
            self.outputs = np.delete(self.outputs, random_idx)

            if test_inputs.shape[0] >= desired_examples:
                break

        train_reshaped = np.reshape(self.outputs, [self.outputs.shape[0], 1])
        test_reshaped = np.reshape(test_outputs, [test_outputs.shape[0], 1])
        return (self.inputs, train_reshaped), (test_inputs, test_reshaped)

    def get_dataset(self):
        self._cook_data()
        return self._split_dataset()


class OffsetMemory(GenericMemory):
    def __init__(self):
        GenericMemory.__init__(self)
        self.inputs = None
        self.outputs = None

    def add_episode(self, input_data, output):
        GenericMemory.add_episode(self, input_data, output)
        # print "Offset episode has added ", input_data, " offset ", output

    def get_dataset(self):
        return clean_values_toone(self.inputs), self.outputs.reshape((-1, 1)).astype(np.float16)


class DepthMemory(GenericMemory):
    def __init__(self):
        GenericMemory.__init__(self)

    def add_pic(self, input_data):
        gray, depth = ppr.rgbd_to_gray_plus_depth(input_data)
        self.add_episode(gray, depth)





def prepare_image(img):
    return img.transpose(2, 0, 1).reshape(1, -1).astype(np.float16) / 255


def prepare_images(img, examples):
    return img.transpose(2, 0, 1).reshape(examples, -1).astype(np.float16) / 255


def clean_values_toone(inputs):
    clean_inputs = np.zeros((inputs.shape[0], 1), np.float32)
    for i in range(0, inputs.shape[0]):
        clean_inputs[i] = reduce(lambda x, y: y + 1 if x > 0.99 else y, inputs[i]) / inputs.shape[0]
    return clean_inputs - 5


def clean_values(inputs):
    clean_inputs = np.zeros(inputs.shape, np.float32)
    for i in range(0, inputs.shape[0]):
        for j in range(0, inputs.shape[1]):
            if inputs[i, j] >= 0.95:
                clean_inputs[i, j] = 0.1
    return clean_inputs


def load_all(name):
    chunks = []
    for fname in os.listdir(params.train_data_folder):
        if fname.startswith(name):
            with open(params.train_data_folder+fname, 'rb') as fh:
                chunks.append(pickle.load(fh))
                print("loaded train example", params.train_data_folder+fname)
    return chunks


def load(path=params.train_data_path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump(obj, path=params.train_data_path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print "Episodes saved", obj.inputs.shape



# load_all("Replay")

r = GenericMemory()

r.add_episode(np.zeros(10), np.zeros(10))
r.add_episode(np.zeros(10), np.zeros(10))