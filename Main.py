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
            return self.model._make_predict_function(arr)

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

class GameGenerator:
    def __init__(self, config, agent):
        self.game = get_game_object()
        self.tree = get_tree(config, agent, self.game)
        self.history = []
        self.policy_targets = []
        self.player_moved_list = []
        self.positions = []

    def run_part1(self):
        return self, self.tree.search_part1()

    def run_part2(self, result):
        self.tree.search_part2(result)
        return self

    def execute_best_move(self):
        temp_move = self.tree.get_temperature_move(self.tree.root)

        self.history.append(temp_move)
        self.policy_targets.append(np.array(self.tree.get_posterior_probabilities()))
        self.player_moved_list.append(self.game.get_turn())
        self.positions.append(np.array(self.game.get_board()))

        self.game.execute_move(temp_move)
        return (self, self.game.is_final())

    def reset_tree(self):
        self.tree.reset_search()
        self.tree.root.board_state = self.game.get_board()

    def get_results(self):
        game_outcome = self.game.get_outcome()
        value_targets = [game_outcome[x] for x in self.player_moved_list]
        return np.array(self.positions), np.array(self.policy_targets), np.array(value_targets)


# Generating data by self-play
def generate_data(game, agent, config, num_sim=100, games=1, num_search=100):
    #tree = get_tree(config, agent, game)

    game_generators = [GameGenerator(config, agent) for _ in range(num_sim)]

    x = []
    y_policy = []
    y_value = []

    while len(game_generators):
        res = [p.apply_async(game_generator.reset_tree) for game_generator in game_generators]
        res = [r.get() for r in res]
        for i in range(num_search):
            res = [p.apply_async(game_generator.run_part1) for game_generator in game_generators]
            res = [r.get() for r in res]
            batch = np.array([result for game_generator, result in res])
            results = agent.predict(batch)
            res = [p.apply_async(res[i].run_part2, (results[i])) for i in range(len(res))]
            game_generators = [r.get() for r in res]
        res = [p.apply_async(game_generator.execute_best_move) for game_generator in game_generators]
        res = [r.get() for r in res]
        game_generators = []
        finished_games = []
        for game_generator, finished in res:
            if finished:
                finished_games.append(game_generator)
                break
            game_generators.append(game_generator)
        game_results = [p.apply_async(game_generator.get_results) for game_generator in finished_games]
        game_results = [r.get for r in game_results]
        for history, policy_targets, value_targets in game_results:
            x += history
            y_policy += policy_targets
            y_value = value_targets
    return np.array(x), np.array(y_policy), np.array(y_value)


# Training AlphaZero by generating data from self-play and fitting the network
def train(game, config, num_filters, num_res_blocks, num_sim=100, epochs=10, games_each_epoch=100,
          batch_size=64, num_train_epochs=1):

    h, w, d = config.board_dims[1:]

    agent = ResNet.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)
    agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))
    agent.summary()

    for epoch in range(epochs):
        x, y_pol, y_val = generate_data(game, agent, config, num_sim=num_sim, games=games_each_epoch)
        print("Epoch")
        print(x.shape)
        raw = agent.predict(x)
        for num in range(len(x)):
            print("targets-predictions")
            print(y_pol[num], y_val[num])
            print(softmax(y_pol[num], raw[0][num]), raw[1][num])

        agent.fit(x=x, y=[y_pol, y_val], batch_size=min(batch_size, len(x)), epochs=num_train_epochs, callbacks=[])
        print("end epoch")
        agent.save_weights("Models/" + Config.name + "/" + str(epoch) + ".h5")
    return agent

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