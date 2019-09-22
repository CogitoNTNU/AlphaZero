import numpy as np

name = "FourInARow"
board_dims = (1, 6, 7, 2)
policy_output_dim = (7)


def NN_output_to_moves(array):
    return [num[0] for num, x in np.ndenumerate(array) if x > 0]


def number_to_move(number):
    return int(number)


def move_to_number(action):
    return int(action)
