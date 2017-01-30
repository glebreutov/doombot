import enemydetector1
import random as rnd
import argparse as ap

from neon.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Uniform
from neon.layers import Conv, Pooling, Affine, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Tanh
import numpy as np
from neon.backends import gen_backend
import bot_params as params
import replay_memory as mem
from enemydetector1 import model, predict

params.batch_size = 64
be = gen_backend(backend='cpu', batch_size=params.batch_size)

dataset = mem.load()


opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9,
                                  stochastic_round=0)

cost = GeneralizedCost(costfunc=CrossEntropyMulti(scale=10))

(X_train, y_train), (X_test, y_test) = dataset.get_dataset()

print X_train.shape, y_train.shape, X_test.shape, y_test.shape
train_set = ArrayIterator(X=X_train, y=y_train, nclass=dataset.nclass, lshape=dataset.shape, make_onehot=False)
test = ArrayIterator(X=X_test, y=y_test, nclass=dataset.nclass, lshape=dataset.shape, make_onehot=False)

callbacks = Callbacks(model, eval_set=test, eval_freq=1,)

model.fit(train_set, optimizer=opt_gdm, num_epochs=2, cost=cost, callbacks=callbacks)
model.save_params(params.weigths_path)


def test_example(i):
    val = predict(X_train[i])
    print "predicted", val, "for", y_train[i]


# for i in range(0, y.shape[0]):
#     val = nn.predict(X[i])
#     if val != 1:
#         print i


test_example(0)
test_example(1)
test_example(2)
test_example(3)
test_example(4)