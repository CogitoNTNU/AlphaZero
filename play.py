# -*- coding: utf-8 -*-

from Main import *
from time import sleep
from Gamerendering import GameRendering
from keras.models import load_model

parser = argparse.ArgumentParser(description='Command line for AZ!')
parser.add_argument("--game", default= "TicTacToe",
                    choices= ["TicTacToe", "FourInARow"], required=False, help= "Choose one of the games from the list")
parser.add_argument("--numSearch", type = int,  default = 500, help = "This is number of searches preformed by MCTS")
args = parser.parse_args()
typeOfGame = args.game
numSearch = args.numSearch
if typeOfGame == "FourInARow":
    game = Gamelogic.FourInARow()
elif typeOfGame == "TicTacToe":
    game = Gamelogic.TicTacToe()

"""get board dimensions and build agent"""
h, w, d = Config.board_dims[1:]
if typeOfGame == "FourInARow":
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5) #Tidligere num_res_blocks=7
elif typeOfGame == "TicTacToe":
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5)

agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))

"""retrieve weights file"""
if typeOfGame == "FourInARow":
    agent.load_weights('Models/FourInARow/b199.h5')
elif typeOfGame == "TicTacToe":
    agent.load_weights('Models/TicTacToe/40.h5')

"""start game-loop"""
rendering = GameRendering(game, agent, Config, numSearch)

