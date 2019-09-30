import numpy as np
import random

class agent0:
    def __init__(self):
        pass

    def predict(self, board_state):
        return [[0.0, 0.45, 0.5, 0.05, 0.0, 0.0, 0.0],0.77]

    def get_info_from_NN(self, board_state):
        return random.uniform(0,1)
