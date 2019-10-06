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
from TicTacToe.Config import policy_output_dim

C_PUCT = math.sqrt(2)


# OBS: when the game is over it the algorithm expects that it is none to move
class Node:
    def __init__(self, game, parent, action, probability=0, t=0, n=0):
        self.parent = parent    #This nodes parent
        self.game = game    #The game played
        self.t = t      #Sum of values from all tree searches through this node
        self.n = n      #Sum of visits from all tree searches through this node
        self.last_action = action   #action from parent board state to current board state
        self.children = []     #List of children nodes
        self.probability = probability  #The probability of choosing this node from the parent node
        if parent:   
            parent.add_child(self)  #adds this node to the parents list of childs
            self.game.execute_move(action)  #executes the move for this node
            self.board_state = self.game.get_board()  #sets the board state for this node
            self.turn = self.game.get_turn()
            self.game.undo_move() #resets the games board state 
        else:
            self.turn = game.get_turn()

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
        self.root = Node(game, None, None)
        self.game = game
        self.root.board_state = start_state
        self.start_state = start_state
        self.agent = agent
        self.T = 1
        self.level = 0

    def reset_search(self):
        self.root = Node(self.game, None, None)
        self.root.board_state = self.game.get_board()
    
    @staticmethod   
    def search_nodechildren_for_state(node, state):
        if node.get_board_state() == state:
            return node
        if not node.is_leaf_node():
            possible_correct = None
            for child in node.children:
                possible_correct = MCTS.search_nodechildren_for_state(child,state)
                if not possible_correct == None:
                    return possible_correct
        
    
    def find_node_given_state(self, state):
        correct = None
        start = self.root
        correct = MCTS.search_nodechildren_for_state(start,state)
        return correct


    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        self.agent = eval

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
    def get_posterior_probabilities(self):
        node = self.root
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
    #TODO ikke lage noder når game er over  
    def search(self):
        game = self.game
        parent = self.root
        while not parent.is_leaf_node():
            best_puct = 0
            for child in parent.children:
                curr_puct = self.PUCT(parent, child)
                if (curr_puct >= best_puct):
                    best_child = child
                    best_puct = curr_puct
            self.level += 1
            parent = best_child
            self.game.execute_move(best_child.last_action)
        result = self.agent.predict(np.array([parent.get_board_state()]))

        if not self.game.is_final():
            valid_moves = game.get_moves_from_board_state(parent.get_board_state())
            for move in valid_moves:
                Node(game, parent, move, result[0][0][move])
            parent.n += 1
            self.back_propagate(parent, result[1][0][0])
            self.level = 0
        else:
            parent.n += 1
            self.back_propagate(parent, result[1][0][0])
            self.level = 0

    def back_propagate(self, node, t):
        game = self.game
        if game.is_final():
            node.t = (game.get_outcome()[node.parent.turn])
        else:
            node.t += t
            node.n += 1
        
        if node.get_parent() is not None:
            game.undo_move()
            self.back_propagate(node.get_parent(), -t)

    def PUCT(self, node, child):
        N = child.n + 1
        sum_N_potential_actions = node.n - 1
        U = C_PUCT * child.probability * math.sqrt(sum_N_potential_actions)/(1+N)
        Q = child.t/N
        return Q + U
    



