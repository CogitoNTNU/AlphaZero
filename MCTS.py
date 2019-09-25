import math
import time
import random
import numpy as np
import random
# from Othello import Gamelogic
from FourInARow import Gamelogic
#import loss
import collections

C_PUCT = math.sqrt(2)
game = Gamelogic()

# OBS: when the game is over it the algorithm expects that it is none to move
class Node:
    def __init__(self, parent, action, t=0, n=0):
        self.parent = parent
        if parent:
            parent.add_child(self)
        self.t = t
        self.n = n
        self.last_action = action
        if parent:
            self.board_state = game.create_game(parent.get_board_state).execute_move(action).get_board_state()
        self.children = []
    
    def get_parent(self):
        return self.parent
    
    def add_child(self, child):
        self.children.append(child)
    
    def is_leaf_node(self):
        if len(self.children) == 0:
            return True
        return False
    
    def get_board_state(self):
        return self.board_state
    
    def get_last_action(self):
        return self.last_action
    
    def get_times_visited(self):
        return self.n
    
    def get_total_values(self):
        return self.t

class MCTS:
    
    def __init__(self, tree, start_state, agent):
        self.tree = Node(None, None)
        self.tree.board_state = start_state
        self.start_state = start_state
        self.agent = agent
        self.T = 1
        self.level = 0

    def reset_search(self):
        self.tree = Node(None, None)
        self.tree.board_state = self.start_state
    
    def find_node_given_state(self, state):
        correct = None
        start = self.tree

        def search_nodechildren_for_state(self, node, state):
            if node.get_board_state() == state:
                return node
            if not node.is_leaf_node():
                possible_correct = None
                for child in node.children:
                    possible_correct = search_nodechildren_for_state(child,state)
                    if not possible_correct == None:
                        return possible_correct
        
        correct = search_nodechildren_for_state(start,state)
        return correct


    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        pass

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self, node_state):
        action_numbers = {}
        node = node_state
        for child in node.children:
            action_numbers[child.last_action] = child.get_times_visited()
        return action_numbers   

    # Returning the prior probabilities of a state, also known as the "raw" NN predictions
    def get_prior_probabilities(self, board_state):
        return agent.predict(board_state)[1].flatten()

    # Returning the posterior search probabilities of the search,
    # meaning that the percentages is calculated by: num_exec/total
    def get_posterior_probabilities(self, board_state):
        node = self.find_node_given_state(board_state)
        tot = 0
        post_prob = {}
        actions = self.get_action_numbers(node)
        for action in actions:
            tot += actions[action]
        for action in actions:
            post_prob[action] = actions[action] / tot
        return post_prob

    # Returning the temperature probabilities calculated from the number of searches for each action
    def get_temperature_probabilities(self, node_state):
        pi = {}
        actions = self.get_action_numbers(node_state)
        for action in actions:
            pi[action] = (actions[action])**(1/T)
        return pi    


    # Returning a random move proportional to the temperature probabilities
    def get_temperature_move(self, node_state):
        pi = self.get_temperature_probabilities(node_state)
        pi_sum = 0
        for value in pi:
            pi_sum = pi_sum + value
        choice = np.random.uniform(0, pi_sum)
        tellesum = 0
        for i in range(0, len(pi)):
            tellesum = tellesum + pi[i]
            if choice < tellesum:
                return i
    
    def evaluate(self, board_state, to_play):
        if to_play != 0:
            value = 1 - self.evaluate(board_state, 0)
        else:
            value = get_info_from_NN(board_state)
            return value

    def get_most_searched_move(self, node_state):
        actions = self.get_action_numbers(node_state)
        most_searched_move = 0
        max = -1    
        for action in actions:
            if actions[action] > max:
                most_searched_move = action
                max = actions[action]
        return most_searched_move

    # Executing MCTS search a "number" times
    def search_series(self, number):
        for _ in range(number):
            self.search()

    # Executing a single MCTS search: Selection-Evaluation-Expansion-Backward pass
    def search(self):
        node = self.tree
        self.best_child = None
        while not node.is_leaf_node():
            best_puct = 0
            for n in node.children:
                curr_puct = self.PUCT(n, n.action)
                if (curr_puct > best_puct):
                    best_child = n
                    best_puct = curr_puct
            self.level += 1
            node = best_child
        node.t = self.agent.predict(node.get_board_state())[0]
        valid_moves = self.game.get_moves(node.get_board_state())
        for move in valid_moves:
            child = Node(node, move)
        self.back_propagate(node)

    def back_propagate(self, node):
        turn = self.level % 2
        if game.create_game(node.get_board_state()).is_final():
            node.t = (game.get_outcome()[turn] + 1) / 2
            (node.get_parent()).n += 1
            self.back_propagate(node.get_parent())
        self.level = 0
        elif node.parent != None:
            (node.get_parent()).t += node.t
            (node.get_parent()).n += 1
            self.back_propagate(node.get_parent())

    def PUCT(self, node_state, action):
        actions = self.get_action_numbers(node_state)

        action_state = None
        for child in node_state.children:
            if child.get_last_action() == action:
                action_state = child

        N = actions[action]
        sum_N_potential_actions = sum(actions.values())
        U = C_PUCT * self.get_prior_probabilities(node_state)*math.sqrt(sum_N_potential_actions)/(1+N)

        Q = action_state.get_total_values()/N

        return Q + U