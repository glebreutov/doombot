import argparse as ap
import random as rand
from vizdoom import Button, GameVariable
import numpy as np
import enemydetector1 as nn
import replay_memory
import bot_params as params
from PIL import Image
import time
import aiming
import bot_params


class ArraySlider:
    def __init__(self, window_w, window_h, stride_x, stride_y, img_shape):
        self.window = ap.Namespace(**{'width': window_w, 'height': window_h})
        self.stride = ap.Namespace(**{'x': stride_x, 'y': stride_y})
        self.img_shape = img_shape

    def slide_over(self, array):
        pos = ap.Namespace(**{'x': 0, 'y': 0})
        while pos is not None:
            if self.self.detector(self.get_subarray(array, pos)):
                return pos
            pos = self.next_pos(array, pos)
        return None

    def next_pos(self, image, pos):
        height = len(image[0])
        width = len(image)
        if pos.x + self.stride.x + self.window.width < width:
            pos.x += self.stride.x
            return pos
        elif pos.y + self.stride.y + self.window.height < height:
            pos.x = 0
            pos.y += self.stride.y
            return pos
        return None

    def get_subarray(self, arr, position):
        return arr[position.y:position.y + self.window.height, position.x:position.x + self.window.width]

    def center_frame(self, offset=0):
        return ap.Namespace(**{'x': int(self.img_shape[1] / 2 - self.window.width / 2 + offset * self.window.width),
                               'y': int(self.img_shape[0] / 2 - self.window.height / 2)})

    def get_whole_line(self, img):
        cf = self.center_frame(0)
        return img[cf.y:cf.y+self.window.height]

    def random_frame(self):
        center = self.center_frame()
        xrange = range(0, center.x - self.window.width / 2)
        l = range(center.x + int(self.window.width * 1.5), self.img_shape[1] - self.window.width)
        xrange.extend(l)

        yrange = range(0, self.img_shape[0] - self.window.height)
        return ap.Namespace(**{'x': rand.choice(xrange),
                               'y': rand.choice(yrange)})


class ShootStat:
    def __init__(self):
        self.shoot_count = 0.
        self.frag_count = 0.00001

    def update_stat(self, hit):
        self.shoot_count += 1
        if hit:
            self.frag_count += 1
        print "frag count", self.frag_count, "shot count", self.shoot_count, "hit percent", self.frag_count/self.shoot_count


def get_possible_frames(img_width, stride, frame_width):
    return [x for x in range(0, img_width - frame_width + 1, stride)]


slider = ArraySlider(window_w=64, window_h=64, stride_x=6, stride_y=6, img_shape=(240, 320, 3))

stat = ShootStat()

#mem = replay_memory.ScreenMemory(1)

nn.load()


def get_offset_from_positon(pos_x):
    return pos_x + slider.window.width / 2


def make_frame(image, pos, frame):
    red = np.array([254, 0, 0])
    for x in range(pos.y, pos.y + frame.height):
        image[x, pos.x] = red
        image[x, pos.x+frame.width] = red

    for x in range(pos.x, pos.x + frame.width):
        image[pos.y, x] = red
        image[pos.y+frame.height, x] = red
    return image



# nn.load()
center_frame = slider.center_frame()

cwd = 0
max_weapon_dev = 100


def make_action(game):

    if game.get_state() is None or game.get_state().game_variables is None:
        return

    if game.get_state().game_variables is not None and game.get_state().game_variables[1] == 0:
        time.sleep(0.2)
        # return

    image_buffer = game.get_state().screen_buffer

    if image_buffer is None:
        return

    x_new = np.zeros((9, 64*64*3), dtype=np.float16)

    for i in range(-4, 4,):
        frame = slider.center_frame(float(i) / 2)
        img = slider.get_subarray(image_buffer, frame)
        x_new[i+4] = replay_memory.prepare_image(img)

    predictions = nn.predict_batch(x_new)

    offset = np.argmax(predictions)

    if predictions[offset] > 0.99:
        mouse_pos = aiming.get_offset(predictions)

        game.make_action([mouse_pos, 0])
        time.sleep(0.3)
        rew = game.make_action([0, 1], 8)
        time.sleep(0.3)
        #time.sleep(0.3)
        #print ammo_before - game.get_state().game_variables[0]

        frag = rew >= 0
        stat.update_stat(frag)
        if frag:
            aiming.it_was_correct(predictions, mouse_pos)
    else:
        game.make_action([rand.randint(-10, 10), 0], 1)


def game_end():
    if aiming.offset_memory.outputs.size % 10 == 0:
        replay_memory.dump(aiming.offset_memory, bot_params.offset_data_path)
    return aiming.offset_memory.outputs.size >= 1000


def save_image(data, name):
    img = Image.fromarray(data, 'RGB')
    img.save(params.train_data_folder + "samples/" + name + '.png')




