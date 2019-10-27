# Importing files from this project
import ResNet
import MCTS
import Files

from FourInARow import Gamelogic
from FourInARow import Config
# from TicTacToe import Gamelogic
# from TicTacToe import Config
#from FourInARow import Gamelogic
#from FourInARow import Config
from keras.optimizers import SGD
from loss import softmax_cross_entropy_with_logits, softmax
#import NN2

# Importing other libraries
import numpy as np

# TODO: game outcome not propagated correctly?

# Creating and returning a tree with properties specified from the input
def get_tree(config, agent, game, dirichlet_noise=True):
    tree = MCTS.MCTS()
    tree.dirichlet_noise = dirichlet_noise
    tree.NN_input_dim = config.board_dims
    tree.policy_output_dim = config.policy_output_dim
    tree.NN_output_to_moves_func = config.NN_output_to_moves
    tree.move_to_number_func = config.move_to_number
    tree.number_to_move_func = config.number_to_move
    tree.set_evaluation(agent)
    tree.set_game(game)
    return tree


# Generating data by self-play
def generate_data(game, agent, config, num_sim=100, games=1):
    tree = get_tree(config, agent, game)

    x = []
    y_policy = []
    y_value = []

    for curr_game in range(games):

        game.__init__()
        # game.execute_move(0)
        # game.execute_move(3)
        # game.execute_move(1)
        # game.execute_move(4)
        # game.execute_move(6)
        # game.execute_move(7)
        history = []
        policy_targets = []
        player_moved_list = []
        positions = []

        while not game.is_final():
            tree.reset_search()
            #tree.root.board_state = game.get_board()
            print("num_sim", num_sim)
            tree.search_series(num_sim)
            # tree.search_series(10)
            state = game.get_state()
            temp_move = tree.get_temperature_move(state)
            print("move:", temp_move)
            print("temp_probs:", tree.get_temperature_probabilities(state))
            history.append(temp_move)
            policy_targets.append(np.array(tree.get_posterior_probabilities(state)))
            print("prior_probs:", tree.get_prior_probabilities(state)) #reshape(1,3,3,2)
            print("pol_targets", policy_targets[-1])
            player_moved_list.append(game.get_turn())
            positions.append(np.array(game.get_board()))

            game.execute_move(temp_move)

        game_outcome = game.get_outcome()
        value_targets = [game_outcome[x] for x in player_moved_list]
        print("val_targets:", value_targets)

        x = x + positions
        y_policy = y_policy + policy_targets
        y_value = y_value + value_targets

        print("History:", history)

    return np.array(x), np.array(y_policy), np.array(y_value)


# Training AlphaZero by generating data from self-play and fitting the network
def train(game, config, num_filters, num_res_blocks, num_sim=400, epochs=200, games_each_epoch=10,
          batch_size=32, num_train_epochs=3):
    h, w, d = config.board_dims[1:]
    # agent, agent1 = NN2.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)
    agent = ResNet.ResNet.build(h, w, d, num_filters, config.policy_output_dim, num_res_blocks=num_res_blocks)
    agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))
    agent.summary()

    # game.__init__()
    # game.execute_move(0)
    # game.execute_move(3)
    # game.execute_move(1)
    # game.execute_move(4)
    # game.execute_move(6)
    # game.execute_move(7)

    # print(agent.predict(game.get_board().reshape(1,3,3,2)))

    for epoch in range(epochs):
        x, y_pol, y_val = generate_data(game, agent, config, num_sim=num_sim, games=games_each_epoch)
        print("Epoch")
        print(x.shape)
        raw = agent.predict(x)
        for i in range(len(x)):
            print("predictions", softmax(y_pol[i], raw[0][i]), raw[1][i])
        agent.fit(x=x, y=[y_pol, y_val], batch_size=min(batch_size, len(x)), epochs=num_train_epochs, callbacks=[])
        print("end epoch")
        if (epoch % 10 == 0):
            agent.save_weights("Models/"+Config.name+"/"+str(epoch)+"_batch.h5")
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

if __name__ == '__main__':
    train(Gamelogic.FourInARow(), Config, 128, 5)
