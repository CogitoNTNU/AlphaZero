# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 08:33:59 2019

@author: Joule
"""

from Main import *
from TicTacToe import Gamelogic
from TicTacToe import Config
from time import sleep

from keras.models import load_model


h, w, d = Config.board_dims[1:]
agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=5)
agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.001, momentum=0.9))


game = Gamelogic.TicTacToe()

agent.load_weights('Models/TicTacToe/40.h5')
print(agent)

#while not game.is_final():
#    if game.get_turn():
#        tree = MCTS.MCTS(game, game.board, agent)
#        tree.search_series(10)
#        predict = tree.get_most_searched_move(tree.root)
#        print(predict)
#        print(agent.predict(game.board.reshape(1,3,3,2))[0][0][i])
#        game.execute_move(predict)
#    else:
#        player_move = int(input("Choose move: "))
#        game.execute_move(player_move)
#    print(game.board)
#    
    
import importlib
import pygame
import sys

#Endre denne for å endre spill
Game = "TicTacToe"

##importer logikk fra spillet
sys.path.insert(0, './'+Game)#Lar python finne modulene i spillmappen
#importerer modulene som "Config" og "logic"
Config = importlib.import_module("Config", package=Game)
logic = importlib.import_module("Gamelogic", package=Game)

class GameRendering:

    def __init__(self, game):
        pygame.init()
        pygame.font.init()
        self.default_font = pygame.font.get_default_font()
        self.text_size=10
        self.font_color=(0,0,0)
        self.font_renderer = pygame.font.Font(self.default_font, self.text_size)
        self.weights=[0,0,0,0,0,0,0,0,0]

        if Game == "TicTacToe":
            self.tictactoe=True #fikser koden slik at den fungerer på TicTacToe
        else:
            self.tictactoe=False
        self.count = 0
        self.game = game
        self.side_length = 150
        self.line_th = 5
        self.height = Config.board_dims[1]
        self.width = Config.board_dims[2]
        self.image = pygame.image.load("nevraltnett.png")
        self.imagerect = self.image.get_size()
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.move_color = (255, 0, 0)
        self.background_color=(0, 109, 50)
        self.piece_size = 50
        self.screen = pygame.display.set_mode([self.side_length * self.width + self.line_th , max(self.side_length * self.height + self.line_th,0)])#+ self.imagerect[0],self.imagerect[1]

        self._render_background()
        self._render_pieces()
        self._render_possible_moves()
        pygame.display.flip()
        while True:
            try:
                
                self.mouse_pos = pygame.mouse.get_pos()
                if self.game.is_final():
                    self.screen.fill(self.black)
                    myfont = pygame.font.SysFont('Comic Sans MS', 50)
                    if self.game.get_outcome()[0] == 0:
                        textsurface = myfont.render('   Tied  ', False, (255, 255, 255))
                    elif (self.game.get_outcome()[0] == 1 and self.count%2 == 0) or (self.game.get_outcome()[1] == 1 and self.count%2 == 1):
                        textsurface = myfont.render(' AI won! ', False, (255, 255, 255))
                    else:
                        textsurface = myfont.render('Human won', False, (255, 255, 255))
                    self.screen.blit(textsurface,(100,50))
                    
                    myfont = pygame.font.SysFont('Comic Sans MS', 10)
                    textsurface = myfont.render('(Switching sides)', False, (0, 255, 0))
                    self.screen.blit(textsurface,(200,150))
                    pygame.display.flip()
                    print("GAME IS OVER")
                    self.count += 1
    
                    
                    sleep(1)
                    game = Gamelogic.TicTacToe()
                    self.game = game
                    self._render_background()
                    self._render_pieces()
                    self._render_possible_moves()
                    pygame.display.flip()
    
    
                    #self.screen.blit(myfont,[ (self.side_length + self.line_th) // 2 + self.side_length * ((Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,(self.side_length + self.line_th) // 2 + self.side_length * ((Config.move_to_number(move))//self.width) - self.label.get_height() ])
                
                
                elif (self.game.get_turn()+self.count)%2 and not self.game.is_final():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                                          
                        
                        elif pygame.mouse.get_pressed()[0] and self.mouse_pos[0] < self.side_length * self.width + self.line_th and self.mouse_pos[1] < self.side_length * self.height + self.line_th: #Sjekk om musen er innenfor brettet
                            self.execute_move()
                            self._render_background()
                            self._render_pieces()
                            self._render_possible_moves()
                            pygame.display.flip()
                        elif pygame.key.get_pressed()[32]:
                            self.game.undo_move()
                            self._render_background()
                            self._render_pieces()
                            self._render_possible_moves()
                            pygame.display.flip()
                            
                elif not self.game.is_final():
                    tree = MCTS.MCTS(game, game.board, agent)
                    tree.search_series(100)
                    predict = tree.get_most_searched_move(tree.root)
                    game.execute_move(predict)
                    self._render_background()
                    self._render_pieces()
                    self._render_possible_moves()
                    pygame.display.flip()
            except:
                print("Ikke trykk der!")
            

    def _render_background(self):
        self.screen.fill(self.background_color)
        for x in range(self.width + 1):
            pygame.draw.line(self.screen, self.black, [x * self.side_length + 2, 0], [x * self.side_length + 2, self.side_length * self.height + self.line_th-2],
                             self.line_th)
            pygame.draw.line(self.screen, self.black, [0, x * self.side_length + 2], [self.side_length * self.width + self.line_th-2, x * self.side_length + 2],
                             self.line_th)
        #self.screen.blit(self.image,((self.side_length*self.width)+self.line_th,0))

    def _render_pieces(self):
        board = self.game.board
        for x in range(self.width):
            for y in range(self.height):
                if board[self.height - y - 1, x, 0] == 1:
                    pygame.draw.circle(self.screen, self.white,
                                       [(self.side_length + self.line_th) // 2 + self.side_length * x, (self.side_length + self.line_th) // 2 + self.side_length * y], self.piece_size)
                elif board[self.height - y - 1, x, 1] == 1:
                    pygame.draw.circle(self.screen, self.black, [(self.side_length + self.line_th) // 2 + self.side_length * x, (self.side_length + self.line_th) // 2 + self.side_length * y],
                                       self.piece_size)

    def _render_possible_moves(self):
        possible_moves = self.game.get_moves()
        
        self.weights = agent.predict(game.board.reshape(1,3,3,2))[0][0]
        
        for move in possible_moves:
            if self.tictactoe:
                move = (self.width * self.height) - (move)//self.width * self.width + (move) % self.width-self.width
            pygame.draw.circle(self.screen, self.move_color, [
                    (self.side_length + self.line_th) // 2 + self.side_length * (
                                (Config.move_to_number(move)) % self.width),
                    (self.side_length + self.line_th) // 2 + self.side_length * (
                                (Config.move_to_number(move)) // self.width)], self.piece_size)
            #render first line of text
            #self.label = self.font_renderer.render(str(self.weights[move]),1,self.font_color)
            #self.screen.blit(self.label,[ (self.side_length + self.line_th) // 2 + self.side_length * ((Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,(self.side_length + self.line_th) // 2 + self.side_length * ((Config.move_to_number(move))//self.width) - self.label.get_height() ])
            #second line
            #self.label = self.font_renderer.render(str(self.weights[move//self.width][move%self.width][1]),1,self.font_color)
            #self.screen.blit(self.label,[ (self.side_length + self.line_th) // 2 + self.side_length * ((Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,(self.side_length + self.line_th) // 2 + self.side_length * ((Config.move_to_number(move))//self.width) + self.label.get_height() / 2])


    def render(self):
        pass

    def execute_move(self):
        if self.tictactoe:
            self.mouse_pos=(self.mouse_pos[0],self.height*self.side_length-self.mouse_pos[1])
        self.game.execute_move(Config.number_to_move((self.mouse_pos[1] - 2) // self.side_length * self.width + (self.mouse_pos[0] - 2) // self.side_length))#må generaliseres


rendering = GameRendering(game)

