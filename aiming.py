from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum
from neon.layers import Linear, Bias
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
import random
import atexit
import bot_params
import numpy as np
import replay_memory
import enemydetector1


try:
    offset_memory = replay_memory.load(bot_params.offset_data_path)
    print "offsets loaded"
except IOError:
    offset_memory = replay_memory.OffsetMemory()

init_norm = Gaussian(loc=-0.1, scale=0.1)
layers = [Linear(1, init=init_norm), Bias(init=init_norm)]

mlp = Model(layers=layers)

cost = GeneralizedCost(costfunc=SumSquared())
optimizer = GradientDescentMomentum(0.5, momentum_coef=0.9)
try:
    mlp.load_params(bot_params.aim_weights_path)
except IOError:
    print "can't load aiming weights"




def get_offset_manual(predictions):
    #enemy_pos = replay_memory.clean_values_toone(predictions)[0, 0]
    x = 0.
    c = 0
    for i in range(0, len(predictions)):
        if predictions[i] > 0.97:
            x += (i+1)
            c += 1
            if i in (0, 1, 7, 8):
                print "BIG OFFSET", i
    enemy_pos = x /c - 5
    offset = int(10 * enemy_pos)
    print "aiming", enemy_pos, offset
    return offset


aim_hist = 0


def get_offset_random(predictions):
    global aim_hist
    out = random.randrange(-30, 30)

    if abs(aim_hist + out) > 30:
        out = -out


    aim_hist += out
    return out


def get_offset_predicted_naive(predictions):

    (replay_memory.clean_values_toone(predictions)[0]*30)+30

    input_iter = ArrayIterator(X=batch, y=None, make_onehot=False)
    outputs = mlp.get_outputs(input_iter)
    outputs_ = 30 * outputs[0, 0]

    print "outputs", outputs_
    return int(round(outputs_))


def get_offset_predicted(predictions):
    batch = np.zeros((bot_params.batch_size, 1))
    batch[0,0] = replay_memory.clean_values_toone(predictions)[0]

    input_iter = ArrayIterator(X=batch, y=None, make_onehot=False)
    outputs = mlp.get_outputs(input_iter)
    outputs_ = 30 * outputs[0, 0]

    print "outputs", outputs_
    return int(round(outputs_))


def it_was_correct(last_in, last_out):
    # print "naive", replay_memory.clean_values_toone(last_in)[0], last_out
    offset_memory.add_episode(last_in, last_out)

    print "osize", offset_memory.outputs.size
    if offset_memory.outputs is not None and offset_memory.outputs.size % bot_params.batch_size == 0:
        X, y = offset_memory.get_dataset()
        train = ArrayIterator(X=X, y=y, make_onehot=False)
        mlp.fit(train, optimizer=optimizer, num_epochs=1, cost=cost,
                callbacks=Callbacks(mlp))
        mlp.save_params(bot_params.aim_weights_path)


get_offset = get_offset_manual





