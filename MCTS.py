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
        # #######DICTIONARIES#######
        self.search_dict = {}  # [visit cnt, sum. act. val., avg. act val, prior prob.] for each state-act
        self.pos_move_dict = {}  # Possible actions for each state
        self.state_visits = {}  # total visits for each state

        # #######PARAMETERS#######
        self.c_puct = 4  # Used for exploration (larger=>less long term exploration)
        self.c_init = 3  # Used for exploration (larger=>more exploration)

        self.dirichlet_noise = True  # Add dirichlet noise to the prior probabilities of the root
        self.alpha = 1.3  # Dirichlet noise variable
        self.epsilon = 0.5  # The amount of dirichlet noise that is added

        self.temperature = 1.4

        # #######I/O shape for eval.#######
        self.NN_input_dim = None
        self.policy_output_dim = None
        self.NN_output_to_moves_func = None
        self.number_to_move_func = None
        self.move_to_number_func = None

        self.tree_children = [0 for _ in range(61)]

        self.time_1 = 0
        self.time_2 = 0
        self.time_3 = 0
        self.time_4 = 0

        self.search_stack = []
        self.return_vars = None

    def set_seed(self, num):
        self.seed = num + np.random.randint(0, high=100000)
        np.random.seed(self.seed)

    def reset_search(self):
        self.search_dict = {}
        self.pos_move_dict = {}
        self.state_visits = {}

        self.tree_children = [0 for _ in range(61)]

        self.search_stack = []
        self.return_vars = None

    # Setting the game the MCTS will be used on
    def set_game(self, game):
        self.game = game

    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        self.eval = eval

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self):
        state = self._return_one_state(self.game.get_state())
        actions = self.pos_move_dict.get(state)
        return {str(action): self.search_dict.get(action)[0] for action in actions}

    # Returning the prior probabilities of a state, also known as the "raw" NN predictions
    def get_prior_probabilities(self, state):
        prob = np.zeros(self.policy_output_dim)
        for action in self.pos_move_dict[state]:
            prob[self.move_to_number_func(action)] = self.search_dict[str(state) + '-' + str(action)][3]
        return prob

    # Returning the posterior search probabilities of the search,
    # meaning that the percentages is calculated by: num_exec/total
    def get_posterior_probabilities(self, state):
        prob = np.zeros(self.policy_output_dim)
        total_visits = self.state_visits[state]
        for action in self.pos_move_dict[state]:
            prob[self.move_to_number_func(action)] = self.search_dict[str(state) + '-' + str(action)][0] / (
                total_visits)
        return prob

    # Returning the temperature probabilities calculated from the number of searches for each action
    def get_temperature_probabilities(self, state):
        prob = np.zeros(self.policy_output_dim)
        for action in self.pos_move_dict[state]:
            prob[self.move_to_number_func(action)] = self.search_dict[str(state) + '-' + str(action)][0]
        temp = np.power(prob, 1 / self.temperature)
        return temp / temp.sum()

    # Returning a random move proportional to the temperature probabilities
    def get_temperature_move(self, state):
        temp_probs = self.get_temperature_probabilities(state)
        temp_num = np.random.choice(temp_probs.shape[0], 1, p=temp_probs)[0]
        return self.number_to_move_func(temp_num)

    def get_most_searched_move(self, state):
        probs = self.get_posterior_probabilities(state)
        return self.number_to_move_func(probs.argmax())

    # Executing MCTS search a "number" times
    def search_series(self, number):
        for _ in range(number):
            self.search()

    # Executing a single MCTS search: Selection-Evaluation-Expansion-Backward pass
    def search(self):
        # Selection - selecting the path from
        state, action = self._selection()
        self.search_stack.append((state, action))

        # The search traversed an internal node
        if action is not None:
            self.search()

        if self.game.is_final():
            self.return_vars = (None, self.game.get_outcome(), True)
            return None

        # Evaluation
        return self.game.get_board()

    def backpropagate(self, result):
        # print("Search_stack", self.search_stack, self)
        state, action = self.search_stack.pop()
        # Expansion
        if not self.game.is_final():
            value, priors = self._evaluate(result)
            self._expansion(state, priors)
            self.tree_children[len(self.game.history)] += 1
            self.return_vars = (value, [], False)

        while len(self.search_stack):
            state, action = self.search_stack.pop()
            backp_value, outcome, finished = self.return_vars
            # Negating the back-propagated value if it is the opponent to move
            to_move = self.game.get_turn()
            self.game.undo_move()
            moved = self.game.get_turn()
            if finished:
                backp_value = outcome[moved]
            elif to_move is not moved:
                backp_value = -backp_value

            # Backward pass
            self._backward_pass(state, str(state) + '-' + str(action), backp_value)
            self.return_vars = (backp_value, outcome, finished)

    # Selecting the path from the root node to the leaf node and returning the new state and the last action executed
    def _selection(self):

        # Current state of the game
        # state = self._return_one_state(self.game.get_state())
        state = self.game.get_state()

        # state=str(state)
        if state not in self.pos_move_dict:  # New state encountered
            return state, None
        if self.game.is_final():  # Game is over
            return state, None

        # values=[self.PUCT(self.search_dict.get(str(state) + '-' + str(action)),
        #                                    self.state_visits.get(state)) for action in self.pos_move_dict.get(state)]
        # max_value=max(values)
        # best_action=self.pos_move_dict.get(state)[values.index(max_value)]

        best_action = ''
        best_value = None

        # Iterating through all actions to find the best
        for action in self.pos_move_dict.get(state):
            state_action_value = self.PUCT(self.search_dict.get(str(state) + '-' + str(action)),
                                           self.state_visits.get(state))
            if best_value is None or state_action_value > best_value:  # If new best action is found
                best_value = state_action_value
                best_action = action

        now = time.time()
        # Executing action and appending state-action pair to path
        self.game.execute_move(best_action)
        self.time_4 += time.time() - now

        return state, best_action

    # Calculating the value for a state-action pair
    def PUCT(self, args, parent_visits):
        # exploration = math.log((1 + parent_visits + self.c_puct) / self.c_puct) + self.c_init
        exploration = 4
        Q = args[2]
        U = exploration * args[3] * math.sqrt(parent_visits) / (1 + args[0])
        return Q + U

    # Evaluate a state using the evaluation algorithm and returning prior policy probabilities and value
    def _evaluate(self, result, epsilon=0.000001):
        # return 0, {str(act): 1 / len(self.game.get_moves()) for num, act in enumerate(self.game.get_moves())}
        # return random.uniform(-1, 1), {str(act): random.random() for num, act in enumerate(self.game.get_moves())}

        # state = state.reshape(self.NN_input_dim)
        policy, value = result

        policy = policy.flatten()

        legal_moves = np.array(self.game.get_legal_NN_output())
        num_legal_moves = np.sum(legal_moves)

        policy_norm = loss.softmax(legal_moves, policy)

        policy_norm = (policy_norm + epsilon) * legal_moves
        outp = self.NN_output_to_moves_func(policy_norm)
        policy_norm = policy_norm[policy_norm > 0]

        if len(self.state_visits) == 0 and self.dirichlet_noise:
            noise = np.random.dirichlet(np.array([self.alpha for _ in range(num_legal_moves)]), (1))
            noise = noise.reshape(noise.shape[1])
            # print("Adding noise", policy_norm)
            return value, {str(act): (1 - self.epsilon) * policy_norm[num] + self.epsilon * noise[num] for num, act in
                           enumerate(outp)}
        else:
            # print("No noise", policy_norm)
            return value, {str(act): policy_norm[num] for num, act in enumerate(outp)}

    # Initializing a new leaf node
    def _expansion(self, state, priors):
        # Finding all actions
        actions = self.game.get_moves()
        self.pos_move_dict[state] = actions
        self.state_visits[state] = 1

        # Initializing each state action pair
        try:
            for action in actions:
                self.search_dict[str(state) + '-' + str(action)] = [0, 0, 0, priors[str(action)]]
        except:
            print("123")

    # Updating a single node in the tree
    def _backward_pass(self, state, state_action, value):
        state_action_values = self.search_dict.get(state_action)
        self.search_dict[state_action] = [state_action_values[0] + 1,
                                          state_action_values[1] + value,
                                          (state_action_values[1] + value) / (state_action_values[0] + 1),
                                          state_action_values[3]]
        self.state_visits[state] = self.state_visits.get(state) + 1


"""
game = Gamelogic.TicTacToe()
tree = MCTS()
tree.set_game(game)

now = time.time()
for _ in range(6000):
    # print(tree.tree_children)
    tree.search()
print('tot:', time.time() - now)
print(tree.time_1, tree.time_2, tree.time_3, tree.time_4)

def stringify(arr):
    a = [str(arr)]
    return a[0]


arr = np.arange(18).reshape(1, 3, 3, 2)
now = time.time()
for _ in range(8000):
    [stringify(np.flip(arr, -1))]

print(time.time() - now)
"""
