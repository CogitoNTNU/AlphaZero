import numpy as np
import random

class agent0:
    def __init__(self):
        pass

    def predict(self, board_state):
        return [[0.1, 0.4, 0.4, 0.2, 0.1, 0.05, 0.12], 0.77]

    def get_info_from_NN(self, board_state):
        return random.uniform(0,1)
