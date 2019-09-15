import math
import time
import random
import numpy as np
# from Othello import Gamelogic
from TicTacToe import Gamelogic
import loss
import collections


# OBS: when the game is over it the algorithm expects that it is none to move


class MCTS:
    def __init__(self):
        pass

    def reset_search(self):
        pass

    # Setting the game the MCTS will be used on
    def set_game(self, game):
        pass

    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        pass

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self):
        pass

    # Returning the prior probabilities of a state, also known as the "raw" NN predictions
    def get_prior_probabilities(self, state):
        pass

    # Returning the posterior search probabilities of the search,
    # meaning that the percentages is calculated by: num_exec/total
    def get_posterior_probabilities(self, state):
        pass

    # Returning the temperature probabilities calculated from the number of searches for each action
    def get_temperature_probabilities(self, state):
        pass

    # Returning a random move proportional to the temperature probabilities
    def get_temperature_move(self, state):
        pass

    def get_most_seached_move(self, state):
        pass

    # Executing MCTS search a "number" times
    def search_series(self, number):
        pass

    # Executing a single MCTS search: Selection-Evaluation-Expansion-Backward pass
    def search(self):
        pass
