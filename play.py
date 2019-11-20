# -*- coding: utf-8 -*-

from Main import *
from time import sleep
from Gamerendering import GameRendering
from keras.models import load_model
import os

opponents = ["Models for TicTacToe: "]
opponents_TicTacToe = []
opponents_FourInARow = []
for opponent in os.listdir("Models/TicTactoe/"):
    opponents.append(opponent)
    opponents_TicTacToe.append(opponent)
opponents.append("Models for FourInARow: ")
for opponent in os.listdir("Models/FourInARow"):
    opponents.append(opponent)
    opponents_FourInARow.append(opponent)


parser = argparse.ArgumentParser(description='Command line for AZ!')
parser.add_argument("--game", default= "TicTacToe",
                    choices= ["TicTacToe", "FourInARow"], required=False, help= "Choose one of the games from the list")
parser.add_argument("--numSearch", type = int,  default = 2, help = "This is number of searches preformed by MCTS")
parser.add_argument("--opponent", type = str, default= "4r_7", choices = opponents ,
                    help = "Choose the agent you want to play against.")



args = parser.parse_args()
typeOfGame = args.game
numSearch = args.numSearch
opponent = args.opponent


i = 0
while i == 0:
    if typeOfGame == "TicTacToe":
        for o in opponents_TicTacToe:
            if o == opponent:
                i = 1
    if typeOfGame == "FourInARow":
        if opponent == "4r_7":
            opponent = "4r_299.h5"
        for o in opponents_FourInARow:
            if o == opponent:
                i = 1
    if i == 0:
        print("You haven't chosen a opponent from the list, please try again")
        if typeOfGame == "TicTacToe":
            print(opponents_TicTacToe)
        elif typeOfGame == "FourInARow":
            print(opponents_FourInARow)
        opponent = input("Choose the agent you want to play against: ")
        print(opponent)


if typeOfGame == "FourInARow":
    game = Gamelogic.FourInARow()
elif typeOfGame == "TicTacToe":
    game = Gamelogic.TicTacToe()

"""get board dimensions and build agent"""
h, w, d = Config.board_dims[1:]
if typeOfGame == "FourInARow":
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=7)
elif typeOfGame == "TicTacToe":
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=7)


agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))

"""retrieve weights file"""
if typeOfGame == "FourInARow":
    try:
        agent.load_weights(f'Models/FourInARow/{opponent}')
    except:
        agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5)
        agent.load_weights(f'Models/FourInARow/{opponent}')
elif typeOfGame == "TicTacToe":
    try:
        agent.load_weights(f'Models/TicTacToe/{opponent}')
    except:
        agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5)
        agent.load_weights(f'Models/TicTacToe/{opponent}')
# Some of the TicTacToe models have only 5 res_blocks, therefore you have to try with 7 and with 5.
"""start game-loop"""

rendering = GameRendering(game, agent, Config, numSearch)

