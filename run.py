#!/usr/bin/env python
#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
import sys
sys.path.append('ml-bot/')

from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

import nnshoot as nn
from time import sleep
from time import time


# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will work. Note that the most recent changes will add to previous ones.
#game.load_config("../../examples/config/basic.cfg")

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
#game.set_vizdoom_path("../../bin/vizdoom")
game.set_doom_game_path("scenarios/freedoom2.wad")
game.set_doom_scenario_path("scenarios/basic.wad")
game.set_doom_map("map01")
game.set_screen_resolution(ScreenResolution.RES_320X240)
game.set_screen_format(ScreenFormat.RGB24)
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)
game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
game.add_available_button(Button.ATTACK)
game.add_available_game_variable(GameVariable.AMMO2)
game.add_available_game_variable(GameVariable.ATTACK_READY)
game.set_episode_timeout(200)
game.set_episode_start_time(10)
game.set_window_visible(True)
game.set_sound_enabled(False)
game.set_living_reward(-1)
game.set_mode(Mode.PLAYER)
game.init()


# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.
#actions = [[True,False,False],[False,True,False],[False,False,True]]

# Run this many episodes
episodes = 1

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup.
sleep_time = 0.5

while True:

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():
        # if game.get_state().image_buffer is None:
        #
        #     break
        nn.make_action(game)

        # Prints state's game variables.

    if nn.game_end():
        break




# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
