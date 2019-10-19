# Importing files from this project
import ResNet
import MCTS
import Files

# from Othello import Gamerendering
# from Othello import Gamelogic
#from TicTacToe import Gamelogic
#from TicTacToe import Config
from FourInARow import Gamelogic
from FourInARow import Config
from keras.optimizers import SGD
from loss import softmax_cross_entropy_with_logits, softmax

import os

# Importing other libraries
import numpy as np
#import multiprocess
#from multiprocess.managers import BaseManager
import tensorflow as tf
#from pathos.multiprocessing import ProcessingPool as Pool

# Creating and returning a tree with properties specified from the input
def get_tree(config, game, dirichlet_noise=True):
    tree = MCTS.MCTS(game, game.get_board(), None, config)
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
    return Gamelogic.FourInARow()


class GameGenerator:
    def __init__(self, config):
        self.game = get_game_object()
        self.tree = get_tree(config, self.game)
        self.history = []
        self.policy_targets = []
        self.player_moved_list = []
        self.positions = []

    def run_part1(self):
        return self.tree.search_part1()

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
        return self, self.game.is_final()

    def reset_tree(self):
        self.tree.reset_search()
        self.tree.root.board_state = self.game.get_board()

    def get_results(self):
        game_outcome = self.game.get_outcome()
        value_targets = [game_outcome[x] for x in self.player_moved_list]
        return np.array(self.positions), np.array(self.policy_targets), np.array(value_targets)


# Generating data by self-play
def generate_data(game, agent, config, num_sim=100, games=1, num_search=130):
    #tree = get_tree(config, agent, game)

    game_generators = [GameGenerator(config) for _ in range(num_sim)]

    x = []
    y_policy = []
    y_value = []

    while len(game_generators):
        res = [game_generator.reset_tree() for game_generator in game_generators]
        for i in range(num_search):
            res = [game_generator.run_part1() for game_generator in game_generators]
            batch = np.array(res)
            results = agent.predict(batch)
            #print(np.array([results[0][0], results[1][0]]))
            #print(np.array([results[0][i], [results[1][i]]])[0][1])
            res = [game_generators[i].run_part2(np.array([results[0][i], [results[1][i]]])) for i in range(len(res))]
            #print("har gjort 100 søk")
        res = [game_generator.execute_best_move() for game_generator in game_generators]
        game_generators = []
        finished_games = []
        for game_generator, finished in res:
            if finished:
                finished_games.append(game_generator)
                continue
            game_generators.append(game_generator)
        game_results = [game_generator.get_results() for game_generator in finished_games]
        for history, policy_targets, value_targets in game_results:
            x.append(history)
            y_policy.append(policy_targets)
            y_value.append(value_targets)
        #with open("test.txt", "a") as f:
        #    f.write("antall spill igjen: " + str(len(game_generators)) + "\n")
    return np.concatenate(x, axis=0), np.concatenate(y_policy, axis=0), np.concatenate(y_value, axis=0)


# Training AlphaZero by generating data from self-play and fitting the network
def train(game, config, num_filters, num_res_blocks, num_sim=400, epochs=1000000, games_each_epoch=10,
          batch_size=32, num_train_epochs=10):
    h, w, d = config.board_dims[1:]
    # agent, agent1 = NN2.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)
    agent = ResNet.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)
    agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))
    #agent.summary()

    # game.__init__()
    # game.execute_move(0)
    # game.execute_move(3)
    # game.execute_move(1)
    # game.execute_move(4)
    # game.execute_move(6)
    # game.execute_move(7)

    # print(agent.predict(game.get_board().reshape(1,3,3,2)))
    import time
    for epoch in range(epochs):
        x, y_pol, y_val = generate_data(game, agent, config, num_sim=num_sim, games=games_each_epoch)
        #print("Epoch")
        #print(x.shape)
        #raw = agent.predict(x)
        #for num in range(len(x)):
        #    print("targets-predictions")
        #    print(y_pol[num], y_val[num])
        #    print(softmax(y_pol[num], raw[0][num]), raw[1][num])
        agent.fit(x=x, y=[y_pol, y_val], batch_size=min(batch_size, len(x)), epochs=num_train_epochs, callbacks=[])
        #print("end epoch")
        #agent.save_weights("Models/" + Config.name + "/" + str(epoch) + ".h5")
        agent.save_weights("Models/"+Config.name+"/"+str(epoch)+"_"+str(time.time())+".h5")
    return agent

def choose_best_legal_move(legal_moves, y_pred):
    best_move = np.argmax(y_pred)
    print("Best move", best_move)
    if (y_pred[best_move] == 0):
        return None
    if best_move in legal_moves:
        return best_move
    else:
        y_pred[best_move] = 0
        return choose_best_legal_move(legal_moves, y_pred)


train(Gamelogic.FourInARow(), Config, 128, 5)
