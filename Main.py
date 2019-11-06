# Importing files from this project
import ResNet
import MCTS
import Files

from TicTacToe import Gamelogic
from TicTacToe import Config
#from FourInARow import Gamelogic
#from FourInARow import Config
from keras.optimizers import SGD
from loss import softmax_cross_entropy_with_logits, softmax


#Importing other libraries
import numpy as np

# Creating and returning a tree with properties specified from the input
def get_tree(config, agent, game, dirichlet_noise=True):
    tree = MCTS.MCTS(game, game.get_board(), agent, config)
    return tree

# Generating data by self-play
def generate_data(game, agent, config, num_sim=100, games=1):
    tree = get_tree(config, agent, game)

    x = []
    y_policy = []
    y_value = []

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


#train(Gamelogic.TicTacToe(), Config, 128, 5)
