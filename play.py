# -*- coding: utf-8 -*-

from Main import *
from time import sleep
from Gamerendering import GameRendering
from keras.models import load_model

parser = argparse.ArgumentParser(description='Command line for AZ!')
parser.add_argument("--game", default= "TicTacToe",
                    choices= ["TicTacToe", "FourInARow"], required=False, help= "Choose one of the games from the list")
parser.add_argument("--numSearch", type = int,  default = 500, help = "This is number of searches preformed by MCTS")
parser.add_argument("--opponent", type = str, default= "4r_7", help = "Choose the agent you want to play against")

args = parser.parse_args()
typeOfGame = args.game
numSearch = args.numSearch
opponent = args.opponent
if typeOfGame == "FourInARow":
    game = Gamelogic.FourInARow()
    opponent = "10_2_20"
elif typeOfGame == "TicTacToe":
    game = Gamelogic.TicTacToe()

"""get board dimensions and build agent"""
h, w, d = Config.board_dims[1:]
if typeOfGame == "FourInARow":
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=10)
elif typeOfGame == "TicTacToe":
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=10)


agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))

"""retrieve weights file"""
if typeOfGame == "FourInARow":
    agent.load_weights(f'Models/FourInARow/{opponent}.h5')
elif typeOfGame == "TicTacToe":
    try:
        agent.load_weights(f'Models/TicTacToe/{opponent}.h5')
    except:
        agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5)
        agent.load_weights(f'Models/TicTacToe/{opponent}.h5')
# Some of the TicTacToe models have only 5 res_blocks, therefore you have to try with 7 and with 5.
"""start game-loop"""

rendering = GameRendering(game, agent, Config, numSearch)

