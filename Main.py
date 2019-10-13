# Importing files from this project
import ResNet
import MCTS
import Files

# from Othello import Gamerendering
# from Othello import Gamelogic
from TicTacToe import Gamelogic
from TicTacToe import Config
from keras.optimizers import SGD
from loss import softmax_cross_entropy_with_logits, softmax
import os

# Importing other libraries
import numpy as np
import multiprocess
from multiprocess.managers import BaseManager
from multiprocess import Lock
import tensorflow as tf
#from pathos.multiprocessing import ProcessingPool as Pool

class KerasModel():
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        self.mutex = Lock()
        self.model = None

    def initialize(self, h, w, d, num_filters, config, num_res_blocks):
        self.model = ResNet.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)
        self.model.compile(loss = [softmax_cross_entropy_with_logits, 'mean_squared_error'], optimizer=SGD(lr=0.0005, momentum=0.9))

    def predict(self, arr):
        with self.mutex:
            with self.graph.as_default():
                with self.session.as_default():
                    return self.model.predict(np.array(arr))

    def train(self, x, y_pol, y_val, batch_size, num_train_epochs):
        with self.mutex:
            self.model.fit(x=x, y=[y_pol, y_val], batch_size=batch_size, epochs=num_train_epochs, callbacks=[])

class KerasManager(BaseManager):
    pass

KerasManager.register('KerasModel', KerasModel)

def work(kerasmodel, h, w, d, num_filters, config, num_res_blocks):
    kerasmodel.initialize(h, w, d, num_filters, config, num_res_blocks)

# Creating and returning a tree with properties specified from the input
def get_tree(config, agent, game, dirichlet_noise=True):
    tree = MCTS.MCTS(game, game.get_board(), agent)
    #tree.dirichlet_noise = dirichlet_noise
    #tree.NN_input_dim = config.board_dims
    #tree.policy_output_dim = config.policy_output_dim
    #tree.NN_output_to_moves_func = config.NN_output_to_moves
    #tree.move_to_number_func = config.move_to_number
    #tree.number_to_move_func = config.number_to_move
    #tree.set_evaluation(agent)
    #tree.set_game(game)
    return tree
    
def get_game_object():
    return Gamelogic.TicTacToe()

def generate_game(config, agent):
    game = get_game_object()
    tree = get_tree(config, agent, game)

    history = []
    policy_targets = []
    player_moved_list = []
    positions = []

    while not game.is_final():
        tree.reset_search()
        tree.root.board_state = game.get_board()
        tree.search_series(2)

        state = game.get_state()
        temp_move = tree.get_temperature_move(tree.root)

        history.append(temp_move)
        policy_targets.append(np.array(tree.get_posterior_probabilities()))
        player_moved_list.append(game.get_turn())
        positions.append(np.array(game.get_board()))

        game.execute_move(temp_move)
    return positions, policy_targets, value_targets

p = multiprocess.Pool(4)

def a(config, agent):
    return [1,1,1]

# Generating data by self-play
def generate_data(game, agent, config, num_sim=100, games=1):
    #tree = get_tree(config, agent, game)

    res = [p.apply_async(generate_game, (config, agent)) for i in range(num_sim)]
    res = [r.get() for r in res]
    x = [i for arr in res for i in arr[0]] 
    y_policy = [i for arr in res for i in arr[1]]
    y_value = [i for arr in res for i in arr[2]] 
    return np.array(x), np.array(y_policy), np.array(y_value)


# Training AlphaZero by generating data from self-play and fitting the network
def train(game, config, num_filters, num_res_blocks, num_sim=100, epochs=10, games_each_epoch=100,
          batch_size=64, num_train_epochs=1):

    h, w, d = config.board_dims[1:]
    with KerasManager() as manager:
        print('Main', os.getpid())
        kerasmodel = manager.KerasModel()
        work(kerasmodel, h, w, d, num_filters, config, num_res_blocks)

        for epoch in range(epochs):
            x, y_pol, y_val = generate_data(game, kerasmodel, config, num_sim=num_sim, games=games_each_epoch)
            print(x)
            print(len(x))
            kerasmodel.train(x, y_pol, y_val, batch_size, num_train_epochs)

    return kerasmodel

def choose_best_legal_move(legal_moves, y_pred):
    best_move = np.argmax(y_pred)
    print("Best move", best_move)
    if(y_pred[best_move] == 0):
        return None
    if best_move in legal_moves:
        return best_move
    else:
        y_pred[best_move] = 0
        print(y_pred)
        return choose_best_legal_move(legal_moves, y_pred)


train(Gamelogic.TicTacToe(), Config, 128, 4)