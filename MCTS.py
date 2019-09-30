import math
import time
import random
import numpy as np
import random
# from Othello import Gamelogic
from TicTacToe.Config import policy_output_dim
from TicTacToe.Gamelogic import TicTacToe
from FourInARow.Gamelogic import FourInARow
from FourInARow.Config import policy_output_dim

#import loss
import collections
from FakeNN import agent0

C_PUCT = math.sqrt(2)


# OBS: when the game is over it the algorithm expects that it is none to move
class Node:
    def __init__(self, game, parent, action, probability=0, t=0, n=0):
        self.parent = parent
        self.game = game
        self.t = t
        self.n = n
        self.last_action = action
        self.children = []
        self.probability = probability
        if parent:
            parent.add_child(self)
        if parent:
            self.board_state = np.flip(self.game.create_game(parent.get_board_state()).execute_move(action).get_board(), -1)

    def get_parent(self):
        return self.parent
    
    def add_child(self, child):
        self.children.append(child)
    
    def is_leaf_node(self):
        if len(self.children) == 0:
            return True
        return False
    
    def get_board_state(self):
        return np.copy(self.board_state)
    
    def get_last_action(self):
        return self.last_action
    
    def get_times_visited(self):
        return self.n
    
    def get_total_values(self):
        return self.t

class MCTS:
    
    def __init__(self, game, start_state, agent):
        self.tree = Node(game, None, None)
        self.game = game
        self.tree.board_state = start_state
        self.start_state = start_state
        self.agent = agent
        self.T = 1
        self.level = 0

    def reset_search(self):
        self.tree = Node(self.game, None, None)
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
                    possible_correct = self.search_nodechildren_for_state(child,state)
                    if not possible_correct == None:
                        return possible_correct
        
        correct = search_nodechildren_for_state(self, start,state)
        return correct


    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        pass

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self, node):
        action_numbers = {}
        for i in range(policy_output_dim):
            action_numbers[i] = 0
        
        for child in node.children:
            action_numbers[child.last_action] = child.get_times_visited()
        return action_numbers   

    # Returning the prior probabilities of a state, also known as the "raw" NN predictions
    def get_prior_probabilities(self, board_state):
        return self.agent.predict(board_state)[1]

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
    def get_temperature_probabilities(self, node):
        pi = {}
        actions = self.get_action_numbers(node)
        for action in actions:
            pi[action] = (actions[action])**(1/self.T)
        return pi    


    # Returning a random move proportional to the temperature probabilities
    def get_temperature_move(self, node):
        pi = self.get_temperature_probabilities(node)
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
            value = self.agent.predict(board_state)[1]
            return value

    def get_most_searched_move(self, node):
        actions = self.get_action_numbers(node)
        most_searched_move = 0
        max = -1    
        print(actions)
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
        game = self.game
        node = self.tree
        while not node.is_leaf_node():
            best_puct = 0
            for n in node.children:
                curr_puct = self.PUCT(node, n.last_action)
                if (curr_puct >= best_puct):
                    best_child = n
                    best_puct = curr_puct
            self.level += 1
            node = best_child
        result = self.agent.predict(node.get_board_state())
        node.t = result[1]
        if game.create_game(node.get_board_state()).player_turn() == 1:
            node.t = 1-node.t

        valid_moves = game.get_moves_from_board_state(node.get_board_state())
        for move in valid_moves:
            Node(game, node, move, result[0][move])
        node.n += 1
        self.back_propagate(node, node.t)

    def back_propagate(self, node, t):
        turn = self.level % 2
        game = self.game
        if game.get_outcome() != None:
            node.t = (game.get_outcome()[turn] + 1) / 2
            if node.get_parent() is not None:
                (node.get_parent()).n += 1
                self.back_propagate(node.get_parent(), t)
                self.level = 0
        elif node.get_parent() is not None:
            (node.get_parent()).t += t
            (node.get_parent()).n += 1
            self.back_propagate(node.get_parent(), t)

    def PUCT(self, node, action):
        for child in node.children:
            if child.get_last_action() == action:
                action_state = child

        N = action_state.n
        sum_N_potential_actions = node.n - 1
        U = C_PUCT * action_state.probability * math.sqrt(sum_N_potential_actions)/(1+N)

        if N != 0:
            Q = action_state.t/N
        else:
            Q = 100000000
        return Q + U


#game = FourInARow()
#agent = agent0()
#MCTS = MCTS(game, game.get_board(), agent)
#MCTS.search_series(800)
#print(MCTS.get_most_searched_move(MCTS.tree))
#print(MCTS.evaluate(MCTS.tree.get_board_state,0))
#print(MCTS.tree.n)