# Importing files from this project
import ResNet
import MCTS
# <<<<<<< HEAD
import argparse
import Files
from keras.optimizers import SGD


# import Config and Gamelogic, from the game you want to train/play

parser = argparse.ArgumentParser(description='Command line for AZ!')
parser.add_argument("--game", default= "TicTacToe",
                    choices= ["TicTacToe", "FourInARow"], required= False, help= "Choose one of the games from the list")
parser.add_argument("--numSearch", type = int,  default = 100, help = "This is number of searches preformed by MCTS")
parser.add_argument("--opponent", type = str, default= "4r_7", help = "Choose the agent you want to play against")

args = parser.parse_args()
typeOfGame = args.game

if typeOfGame == "TicTacToe":
    print("Skal spille TicTacToe")
    from TicTacToe import Config
    from TicTacToe import Gamelogic
elif typeOfGame == "FourInARow":
    print("Skal spille FourInARow")
    from FourInARow import Config
    from FourInARow import Gamelogic


from loss import softmax_cross_entropy_with_logits, softmax


#Importing other libraries
import numpy as np

# Creating and returning a tree with properties specified from the input
def get_tree(config, agent, game, dirichlet_noise=True):
    tree = MCTS.MCTS(game, game.get_board(), agent, config)
    return tree

# =======
# import Multiprocessing
#
# import time
#
# # from TicTacToe import Gamelogic
# # from TicTacToe import Config
# from FourInARow import Gamelogic
# from FourInARow import Config
#
# # from keras.optimizers import SGD
# # from loss import softmax_cross_entropy_with_logits, softmax
#
# import numpy as np
#
#
# # Creating and returning a tree with properties specified from the input
# def get_tree(config, agent, game, dirichlet_noise=True, seed=0):
#     tree = MCTS.MCTS()  # (game, game.get_board(), None, config)
#     tree.dirichlet_noise = dirichlet_noise
#     tree.NN_input_dim = config.board_dims
#     tree.policy_output_dim = config.policy_output_dim
#     tree.NN_output_to_moves_func = config.NN_output_to_moves
#     tree.move_to_number_func = config.move_to_number
#     tree.number_to_move_func = config.number_to_move
#     tree.set_evaluation(agent)
#     tree.set_game(game)
#     # print("setting seed", seed)
#     tree.set_seed(seed)
#     return tree
#
#
# def get_game_object():
#     return Gamelogic.FourInARow()
#
#
# class GameGenerator:
#     def __init__(self, config, agent, seed=0):
#         self.game = get_game_object()
#         self.tree = get_tree(config, agent, self.game, seed=seed)
#         self.history = []
#         self.policy_targets = []
#         self.player_moved_list = []
#         self.positions = []
#
#     def run_part1(self):
#         return self.tree.search()
#
#     def run_part2(self, result):
#         self.tree.backpropagate(result)
#         return self
#
#     def execute_best_move(self):
#         state = self.game.get_state()
#         temp_move = self.tree.get_temperature_move(state, len_hist=len(self.history))
#         # print(self.tree.get_prior_probabilities(state))
#         # print("temp move", temp_move, self.tree.seed)
#         self.history.append(temp_move)
#         self.policy_targets.append(np.array(self.tree.get_posterior_probabilities(state)))
#         self.player_moved_list.append(self.game.get_turn())
#         self.positions.append(np.array(self.game.get_board()))
#         self.game.execute_move(temp_move)
#         return self, self.game.is_final()
#
#     def reset_tree(self):
#         self.tree.reset_search()
#         # self.tree.root.board_state = self.game.get_board()
#
#     def get_results(self):
#         game_outcome = self.game.get_outcome()
#         value_targets = [game_outcome[x] for x in self.player_moved_list]
#         return self.history, self.positions, self.policy_targets, value_targets
#
#
# >>>>>>> newMCTS
# Generating data by self-play
def generate_data(res_dict, config1, num_games_each_process, num_search, num_process, name_weights, seeds):
    # print("Starting", num_process)
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    # print("_a_")
    import ResNet as ResNet_p
    # print("_a_")
    from keras.backend.tensorflow_backend import set_session
    # print("_a_")
    from keras.optimizers import SGD
    # print("_j_")
    from loss import softmax_cross_entropy_with_logits, softmax
    # print("_i_")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.015)
    # print("_h_")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # print("_g_")
    set_session(sess)
    # print("_f_")

    h, w, d = config1.board_dims[1:]
    # print("_e_")
    agent = ResNet_p.ResNet.build(h, w, d, 128, config1.policy_output_dim, num_res_blocks=10)
    # print("_d_")
    agent.load_weights(name_weights)
    # print("_c_")

    game_generators = [GameGenerator(config1, agent, seed=seeds[i]) for i in range(num_games_each_process)]
    # print("_b_")

    x = []
    y_policy = []
    y_value = []
# <<<<<<< HEAD

    for _ in range(games):

        game.__init__()
        history = []
        policy_targets = []
        player_moved_list = []
        positions = []

        while not game.is_final():
            tree.reset_search()
            tree.root.board_state = game.get_board()
            tree.search_series(num_sim)
            temp_move = tree.get_temperature_move(tree.root)
            history.append(temp_move)
            policy_targets.append(np.array(tree.get_posterior_probabilities()))
            if typeOfGame == "FourInARow":
                (tree.get_prior_probabilities(game.get_board().reshape(1, 6, 7, 2)))
            if typeOfGame == "TicTacToe":
                (tree.get_prior_probabilities(game.get_board().reshape(1,3,3,2)))
            player_moved_list.append(game.get_turn())
            positions.append(np.array(game.get_board()))

            game.execute_move(temp_move)
            print("________________")
            game.print_board()
            print("________________")

        game_outcome = game.get_outcome()
        if game_outcome == [1, -1]:
            print("X vant")
        elif game_outcome == [-1, 1]:
            print ("O vant")
        else:
            print("Uavgjort")
        
        value_targets = [game_outcome[x] for x in player_moved_list]

        x = x + positions
        y_policy = y_policy + policy_targets
        y_value = y_value + value_targets


    return np.array(x), np.array(y_policy), np.array(y_value)


# Training AlphaZero by generating data from self-play and fitting the network
def train(game, config, num_filters, num_res_blocks, num_sim=125, epochs=50, games_each_epoch=10,
          batch_size=32, num_train_epochs=10):
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
        agent.save_weights("Models/"+Config.name+"/"+str(epoch)+".h5")
    return agent
# =======
    print("Ready to play", num_process)

    predicted = {}

    while len(game_generators):
        # print("test1")
        res = [game_generator.reset_tree() for game_generator in game_generators]
        print("len_dict", len(predicted.keys()))
        for i in range(num_search):
            # print("test2")
            res = [game_generator.run_part1() for game_generator in game_generators]
            # print("test3")
            to_predict = []
            to_predict_generators = []
            no_predict_generators = []
            predicted_generators = []
            predicted_states = []
            for l in range(len(res)):
                # print("test4")
                if res[l] is not None:
                    if np.array_str(res[l]) in predicted:
                        predicted_generators.append(game_generators[l])
                        predicted_states.append(res[l])
                    else:
                        # print("To predict", res[l])
                        # print("Stack:", game_generators[l].tree.search_stack, game_generators[l].tree, game_generators[l])
                        to_predict.append(res[l])
                        to_predict_generators.append(game_generators[l])
                        predicted[np.array_str(res[l])] = None
                else:
                    no_predict_generators.append(game_generators[l])
                # print("test5")
            # print("test6")
            if len(to_predict):
                # print("to_predict", len(to_predict))
                batch = np.array(to_predict)
                results = agent.predict(batch)
                for j in range(len(to_predict)):
                    predicted[np.array_str(to_predict[j])] = [results[0][j], results[1][j][0]]
                # print("result", results)

            # print("test7")
            [predicted_generators[j].run_part2(np.array(predicted[np.array_str(predicted_states[j])])) for j in
             range(len(predicted_generators))]
            [to_predict_generators[j].run_part2(np.array([results[0][j], results[1][j][0]])) for j in
             range(len(to_predict_generators))]
            # print("test8")
            [no_predict_generators[i].run_part2(None) for i in range(len(no_predict_generators))]
            # print("test9")

        print("LEN", len(game_generators))

        res = [game_generator.execute_best_move() for game_generator in game_generators]
        game_generators = []
        finished_games = []
        for game_generator, finished in res:
            if finished:
                finished_games.append(game_generator)
                continue
            game_generators.append(game_generator)
        game_results = [game_generator.get_results() for game_generator in finished_games]
        for moves, history, policy_targets, value_targets in game_results:
            print("moves", moves)
            x.extend(history)
            y_policy.extend(policy_targets)
            y_value.extend(value_targets)

    print("finished", )
    res_dict[str(num_process)] = [x, y_policy, y_value]


# # Generating data by self-play
# def generate_data(game, agent, config, num_sim=100, games=1):
#     tree = get_tree(config, agent, game)
#
#     x = []
#     y_policy = []
#     y_value = []
#
#     for curr_game in range(games):
#
#         game.__init__()
#         history = []
#         policy_targets = []
#         player_moved_list = []
#         positions = []
#
#         while not game.is_final():
#             tree.reset_search()
#             print("num_sim", num_sim)
#             tree.search_series(num_sim)
#             state = game.get_state()
#             temp_move = tree.get_temperature_move(state)
#             print("move:", temp_move)
#             print("temp_probs:", tree.get_temperature_probabilities(state))
#             history.append(temp_move)
#             policy_targets.append(np.array(tree.get_posterior_probabilities(state)))
#             print("prior_probs:", tree.get_prior_probabilities(state)) #reshape(1,3,3,2)
#             print("pol_targets", policy_targets[-1])
#             player_moved_list.append(game.get_turn())
#             positions.append(np.array(game.get_board()))
#
#             game.execute_move(temp_move)
#
#         game_outcome = game.get_outcome()
#         value_targets = [game_outcome[x] for x in player_moved_list]
#         print("val_targets:", value_targets)
#
#         x = x + positions
#         y_policy = y_policy + policy_targets
#         y_value = y_value + value_targets
#
#         print("History:", history)
#
#     return np.array(x), np.array(y_policy), np.array(y_value)
#
#
# Training AlphaZero by generating data from self-play and fitting the network
# >>>>>>> newMCTS

# Returns the best legal move based on the predictions from
# The neural network
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
# <<<<<<< HEAD


if __name__ == '__main__':
    if typeOfGame == "FourInARow":
        train(Gamelogic.FourInARow(), Config, 128, 7)
    elif typeOfGame == "TicTacToe":
        train(Gamelogic.TicTacToe(), Config, 128, 4)
# =======
# >>>>>>> newMCTS
