from TicTacToe import Gamelogic
import MCTS
import ResNet
from TicTacToe import Config
import Files
import os

game = Gamelogic.TicTacToe()
config = Config

# Creating the NN
h, w, d = game.get_board().shape
agent = ResNet.ResNet.build(h, w, d, 128, config.policy_output_dim, num_res_blocks=5)
agent2 = ResNet.ResNet.build(h, w, d, 128, config.policy_output_dim, num_res_blocks=5)

# Creating the MCTS
tree = MCTS.MCTS(game, game.get_board(), agent)
# tree.dirichlet_noise = False
tree.NN_input_dim = config.board_dims
tree.policy_output_dim = config.policy_output_dim
tree.NN_output_to_moves_func = config.NN_output_to_moves
tree.move_to_number_func = config.move_to_number
tree.number_to_move_func = config.number_to_move
# tree.set_game(game)

for opponent in os.listdir("Models/TicTactoe/"):
    won = 0
    sum = 0
    draw=0
    agent.load_weights("Models/TicTactoe/40.h5")
    agent2.load_weights("Models/TicTactoe/" + opponent)

    for player_start in range(2):
        for start in range(9):
            print("Started\n\n")
            game.__init__()
            game.execute_move(start)
            while not game.is_final():
                game.print_board()
                # input()
                print(game.get_moves())
                print(game.get_legal_NN_output())
                if game.get_turn() == player_start:
                    tree.set_evaluation(agent)
                else:
                    tree.set_evaluation(agent2)
                tree.search_series(2)
                print("post", tree.get_posterior_probabilities())
                print("pri", tree.get_prior_probabilities(game.get_board().reshape(1,3,3,2)))
                game.execute_move(tree.get_most_searched_move(tree.root))
                tree.reset_search()
            won += game.get_outcome()[player_start] == 1
            draw += game.get_outcome()[player_start] == 0
            sum += 1
            print("Finished\n\n")
    print("Result against", opponent, "-", won, "/", draw, "/", sum-won-draw)
