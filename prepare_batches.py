import os
import pickle
import random

import bot_params as params
from replay_memory import DepthMemory
files = os.listdir(params.train_data_folder)

examples_in_file = 64
desired_examples = 64*2

filtered_files = [file for file in files if file.startswith("DepthMemory") and file.endswith(".pickle")]
loaded_files = [None for x in filtered_files]

processed_examples = set()
count = 0


def get_example(pos_tuple):
    global loaded_files
    file_idx = pos_tuple[0]
    ex_idx = pos_tuple[1]
    if loaded_files[file_idx] is None:
        with open(params.train_data_folder + filtered_files[file_idx], "rb") as f:
            loaded_files[file_idx] = pickle.load(f)

    return loaded_files[file_idx].inputs[ex_idx], loaded_files[file_idx].outputs[ex_idx]


current_batch = None


def add_example(inp, out):
    global current_batch
    if current_batch is None:
        current_batch = DepthMemory()

    if current_batch.inputs is not None and current_batch.inputs.shape[0] >= examples_in_file:
        current_batch.save_data("_64")
        current_batch = DepthMemory()

    current_batch.add_episode(inp, out)

while desired_examples > count:
    randex_addr = (random.randint(0, len(filtered_files)-1), random.randint(0, params.num_examples_to_dump)-1)

    if randex_addr not in processed_examples:
        inp, out = get_example(randex_addr)
        add_example(inp, out)
        processed_examples.add(randex_addr)
        count += 1


