train_data_folder = "traindata/"
train_data_path = train_data_folder + 'ReplayMemory.dat'
offset_data_path = train_data_folder + 'OffsetsMemeory.dat'
weigths_path = train_data_folder + "weight1.neon"
aim_weights_path = train_data_folder + "aim_weights.neon"

train_session_num_of_examples = 50000

batch_size = 9

frame_shape = (3, 64, 64)

num_of_strategy_steps = 10
possible_actions_count = 64


num_examples_to_dump = 1000
