from neon.backends import gen_backend
from neon.callbacks import Callbacks
from neon.data import ArrayIterator

import aiming
import bot_params as params
be = gen_backend(backend='cpu', batch_size=params.batch_size)


X, y = aiming.offset_memory.get_dataset()
data_dict = {}
for i in range(0, X.size):
    cx = X[i, 0]
    cy = y[i, 0]
    print cx, cy
    if cx not in data_dict:
        data_dict[cx] = [cy]
    else:
        data_dict[cx].append(cy)

for x1, y1 in data_dict.items():
    print x1, reduce(lambda x, y: x + y, y1) / len(y1)

train = ArrayIterator(X=X, y=y, make_onehot=False)
callbacks = Callbacks(aiming.mlp, train_set=train)

aiming.mlp.fit(train, optimizer=aiming.optimizer, num_epochs=1, cost=aiming.cost, callbacks=callbacks)
aiming.mlp.save_params(params.aim_weights_path)

