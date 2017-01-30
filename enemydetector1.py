import random as rnd
import argparse as ap

from neon.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Uniform
from neon.layers import Conv, Pooling, Affine, GeneralizedCost, SkipNode
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Tanh, Logistic
import numpy as np
from neon.backends import gen_backend
import bot_params as params

import replay_memory as mem

# memory = mem.load()

be = gen_backend(backend='cpu', batch_size=params.batch_size)

init_uni = Uniform(low=-0.1, high=0.1)

bn = True

layers = [Conv((16, 16, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Conv((16, 16, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=1, init=init_uni, activation=Logistic(shortcut=True))]

model = Model(layers=layers)


def load():
    model.load_params(params.weigths_path)


def predict(input_img):
    # model.set_batch_size(1)
    x_new = np.zeros((params.batch_size, input_img.size), dtype=np.float16)
    x_new[0] = mem.prepare_image(input_img)
    inp = ArrayIterator(X=x_new, y=None, nclass=2, lshape=params.frame_shape, make_onehot=False)
    qvalues = model.get_outputs(inp)
    return qvalues[0][0]


def predict_batch(batch):

    inp = ArrayIterator(X=batch, y=None, nclass=2, lshape=params.frame_shape, make_onehot=False)
    return model.get_outputs(inp)

# train()

