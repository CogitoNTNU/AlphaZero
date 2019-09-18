import math
import time
import random
import numpy as np
# from Othello import Gamelogic
from TicTacToe import Gamelogic
#import loss
import collections

C_PUCT = sqrt(2)

# OBS: when the game is over it the algorithm expects that it is none to move
class Node:
    def __init__(self, parent, t=0, n=0, state):
        self.parent = parent
        if parent:
            parent.add_child(self)
        self.t = t
        self.n = n
        self.state = state
        self.children = []
    
    def get_parent(self):
        return self.parent
    
    def add_child(self, child):
        self.children.append(child)
    
    def is_leaf_node(self):
        if len(children) == 0:
            return True
        return False

class MCTS:
    
    def __init__(self, tree, start_state):
        self.tree = Node(None, start_state)
        self.start_state = start_state

    def reset_search(self):
        self.tree = Node(None, start_state)

    # Setting the game the MCTS will be used on
    def set_game(self, game):
        pass

    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        pass

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self, state):
        action_numbers = {}
        node = None
        while 


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

    def get_most_searched_move(self, state):
        actions = get_action_numbers(state)
        most_searched_move = 0
        max = -1
        for action in actions:
            if actions[action] > max:
                most_searched_move = action
                max = actions[action]
        return most_searched_move

    # Executing MCTS search a "number" times
    def search_series(self, number):
        pass

    # Executing a single MCTS search: Selection-Evaluation-Expansion-Backward pass
    def search(self):
        pass

    def PUCT(self, state, action):
        actions = get_action_numbers()
        N = actions[action]

        U = C_PUCT * get_prior_probabilities(state)*
        return 
