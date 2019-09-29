# Importing files from this project
import ResNet
import MCTS
import Files

# from Othello import Gamerendering
# from Othello import Gamelogic
from TicTacToe import Gamelogic
from TicTacToe import Config

# Importing other libraries
import numpy as np


# Creating and returning a tree with properties specified from the input
def get_tree(config, agent, game, dirichlet_noise=True):
    tree = MCTS.MCTS(game.get_board(), agent)
    #tree.dirichlet_noise = dirichlet_noise
    #tree.NN_input_dim = config.board_dims
    #tree.policy_output_dim = config.policy_output_dim
    #tree.NN_output_to_moves_func = config.NN_output_to_moves
    #tree.move_to_number_func = config.move_to_number
    #tree.number_to_move_func = config.number_to_move
    #tree.set_evaluation(agent)
    #tree.set_game(game)
    return tree


# Generating data by self-play
def generate_data(game, agent, config, num_sim=100, games=1000):
    tree = get_tree(config, agent, game)

    x = []
    y_policy = []
    y_value = []

    for curr_game in range(games):

        game.__init__()
        history = []
        policy_targets = []
        player_moved_list = []
        positions = []

        for i in range(num_sim):
            tree.reset_search()
            tree.tree.board_state = game.get_board()
            tree.search(num_sim)

            state = game.get_state()
            temp_move = tree.get_temperature_move(state)

            history.append(temp_move)
            policy_targets.append(np.array(tree.get_posterior_probabilities(state)))
            player_moved_list.append(game.get_turn())
            positions.append(np.array(game.get_board()))

            game.execute_move(temp_move)


        game_outcome = game.get_outcome()
        value_targets = [game_outcome[x] for x in player_moved_list]

        x = x + positions
        y_policy = y_policy + policy_targets
        y_value = y_value + value_targets

    return np.array(x), np.array(y_policy), np.array(y_value)


# Training AlphaZero by generating data from self-play and fitting the network
def train(game, config, num_filters, num_res_blocks, num_sim=100, epochs=100, games_each_epoch=1000,
          batch_size=64, num_train_epochs=1):

    h, w, d = config.board_dims[1:]
    agent = ResNet.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)

    for epoch in range(epochs):
        x, y_pol, y_val = generate_data(game, agent, config, num_sim=num_sim, games=games_each_epoch)
        print(x)
        print(len(x))
        agent.fit(x=x, y=[y_pol, y_val], batch_size=batch_size, epochs=num_train_epochs, callbacks=[])

    return agent


train(Gamelogic.TicTacToe(), Config, 128, 4)
