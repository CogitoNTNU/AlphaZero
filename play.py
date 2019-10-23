# -*- coding: utf-8 -*-

from Main3 import *

#from TicTacToe import Gamelogic
#from TicTacToe import Config
from FourInARow import Gamelogic
from FourInARow import Config
from time import sleep
from Gamerendering import GameRendering
from keras.models import load_model

"""get board dimensions and build agent"""
h, w, d = Config.board_dims[1:]
agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5)
agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))

"""retrieve weights file"""
game = Gamelogic.FourInARow()
agent.load_weights('Models/FourInARow/6_1571773688.6859412.h5')


"""start game-loop"""
rendering = GameRendering(game, agent, Config)

