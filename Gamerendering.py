import pygame
import sys
from time import sleep
import copy


#from MCTS import MCTS
from Main3 import *
"""Weird bug when trying to import MCTS, so had to star import from Main"""


class GameRendering:


    def __init__(self, game, agent, Config):
        """Initialize the pygame"""
        pygame.init()
        pygame.font.init()
        self.default_font = pygame.font.get_default_font()
        self.text_size = 10
        self.font_color = (40,40,40)
        self.font_renderer = pygame.font.Font(self.default_font, self.text_size)
        """Variables"""
        self.game = copy.deepcopy(game)

        self.Config = Config

        self.start_pos = np.copy(game.board)
        self.agent = agent
        self.side_length = 100
        self.line_th = 5
        self.height = self.game.board_dims[1]
        self.width = self.game.board_dims[2]
#        self.image = pygame.image.load("nevraltnett.png")
#        self.imagerect = self.image.get_size()
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.piece_size = self.side_length//3
        self.screen = pygame.display.set_mode([self.side_length * self.width + self.line_th , max(self.side_length * self.height + self.line_th,0)])#+ self.imagerect[0],self.imagerect[1]
        self.weights = [0]*len(self.game.get_moves())   #Set weight to empty array to represent all moves
        self.count = 0 #Switches sides of human and machine
        self.won = 0
        self.tied = 0
        self.lost = 0

        """find what game is given"""
        self.tictactoe = False
        self.fourinarow = False
        if self.game.name == "TicTacToe":
            self.tictactoe=True 
        elif self.game.name == "FourInARow":
            self.fourinarow = True
        
        self.update_screen()
        
        
        """continuosly check for updates"""
        while True:
            self.mouse_pos = pygame.mouse.get_pos()
            if self.game.is_final():
                """show winning move"""
                self.update_screen()
                if self.game.name == "FourInARow" and self.game.get_winning_pieces() is not None:
                    pygame.draw.line(self.screen, self.white,
                                 [(self.side_length + self.line_th) // 2 + self.side_length * self.game.get_winning_pieces()[0][1], (self.side_length + self.line_th) // 2 + self.side_length * (self.height - self.game.get_winning_pieces()[0][0] - 1)],
                                 [(self.side_length + self.line_th) // 2 + self.side_length * self.game.get_winning_pieces()[1][1], (self.side_length + self.line_th) // 2 + self.side_length * (self.height - self.game.get_winning_pieces()[1][0] - 1)],
                                 self.line_th)
                    pygame.display.flip()
                sleep(1)
                sleep(4)

                """Show death screen"""
                self.screen.fill(self.black)
                myfont = pygame.font.SysFont(self.default_font, 50)
                """Who won"""
                if self.game.get_outcome()[0] == 0:
                    winner = myfont.render('Tied', False, (255, 255, 255))
                    self.tied += 1
                elif (self.game.get_outcome()[0] == 1 and self.count%2 == 0) or (self.game.get_outcome()[1] == 1 and self.count%2 == 1):
                    winner = myfont.render('AI won!', False, (255, 255, 255))
                    self.lost += 1
                else:
                    winner = myfont.render('Human won', False, (255, 255, 255))
                    self.won += 1
                self.screen.blit(winner,((self.side_length*self.width + self.line_th) // 2 - winner.get_width()/2 , self.side_length//3 - winner.get_height()//2))
                
                myfont = pygame.font.SysFont('Comic Sans MS', 10)
                switch_side = myfont.render('(Switching sides)', False, (0, 255, 0))
                self.screen.blit(switch_side,((self.side_length*self.width + self.line_th) // 2 - switch_side.get_width()/2, self.side_length//3 - switch_side.get_height()//2 + winner.get_height()))
                
                """Shows the score"""
                myfont = pygame.font.SysFont(self.default_font, 20)
                wtl = myfont.render('Win/Tie/Loss', False, self.white)
                self.screen.blit(wtl,((self.side_length*self.width + self.line_th) // 2 - wtl.get_width()/2, (self.side_length*self.height)//2 - wtl.get_height()//2))
                score = myfont.render(str(self.won)+"-"+str(self.tied)+"-"+str(self.lost), False, self.white)
                self.screen.blit(score,((self.side_length*self.width + self.line_th) // 2 - score.get_width()/2, (self.side_length*self.height)//2 + wtl.get_height()))
                
                pygame.display.flip()
                print("GAME IS OVER")
                self.count += 1 #Switches sides

                sleep(1)    #Catch glitchy graohics
                sleep(4)    #Hold the death screen open
                """clean the board and graphics"""
                self.weights = [0]*len(self.weights)
                self.game.board = np.copy(self.start_pos)   #don't want to change it
                self.game.history = []
                self.update_screen()
            
            elif (self.game.get_turn()+self.count)%2 and not self.game.is_final():    
                """look for human input"""
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or pygame.key.get_pressed()[27]:   # Escape quits the game
                        sys.exit()
                    elif pygame.mouse.get_pressed()[0] and self.mouse_pos[0] < self.side_length * self.width + self.line_th and self.mouse_pos[1] < self.side_length * self.height + self.line_th: #Check if mouse is pressed in board
                        self.execute_move()
                        self.update_screen()
                    elif pygame.key.get_pressed()[8]:   #Backspace to undo move
                        self.game.undo_move()
                        self.game.undo_move()
                        self.update_screen()
            
            elif not self.game.is_final():
                """If machines turn, machine do move"""
                tree = MCTS.MCTS(self.game, self.game.board, self.agent, self.Config)
                if len(self.game.get_moves()) > 1:   # Does not compute first, and last possible move very deeply
                    for searches in range(125):
                        tree.search()
                        if searches%25 == 0:
                            """update weight on screen every 200 search"""
                            self.weights = tree.get_posterior_probabilities()
                            self.update_screen()
                            self.see_valuation()
                else:
                    tree.search_series(100)
                predict = tree.get_most_searched_move(tree.root)
#                print("Stillingen vurderes som: ",self.agent.predict(np.array([self.game.get_board()]))[1])
                self.game.execute_move(predict)
                self.update_screen()
                self.see_valuation()


    def see_valuation(self):
        """see how the nn values different moves on its turn for itself"""
        if self.tictactoe:
            """Has to flip the logic in y direction since it increases down"""
            possible_moves = self.game.get_moves()
            for move in possible_moves:
                self.label = self.font_renderer.render(str(round(self.weights[move],4)),1,self.font_color)
                self.screen.blit(self.label,[ (self.side_length + self.line_th) // 2 + self.side_length * ((self.Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,(self.side_length + self.line_th) // 2 + self.side_length * ((8-self.Config.move_to_number(move))//self.width) - self.label.get_height() ])
                pygame.display.flip()
        elif self.fourinarow:
            possible_moves = self.game.get_moves()
            for move in possible_moves:
                self.label = self.font_renderer.render(str(round(self.weights[move],4)),1,self.font_color)
                self.screen.blit(self.label,[ (self.side_length + self.line_th) // 2 + self.side_length * ((self.Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,(self.side_length // self.height) - self.label.get_height() ])
                pygame.display.flip()


    def _render(self, background_color, line_color, p1_color, p2_color, possible_color):
        """Generic board maker"""

        """Background"""
        self.screen.fill(background_color)
        """Draw board lines
        pygame.draw.line(surface, color, start_pos, end_pos, width)"""
        for line in range(self.width + 1):
            """Vertical lines"""
            pygame.draw.line(self.screen, line_color,
                             [line * self.side_length + 2, 0],
                             [line * self.side_length + 2, self.side_length * self.height + self.line_th-2],
                             self.line_th)
            """Horizontal lines"""
            pygame.draw.line(self.screen, line_color,
                             [0, line * self.side_length + 2],
                             [self.side_length * self.width + self.line_th-2, line * self.side_length + 2],
                             self.line_th)
            
        """Render pieces"""
        board = self.game.board
        for x in range(self.width):
            for y in range(self.height):
                if board[self.height - y - 1, x, 0] == 1:
                    pygame.draw.circle(self.screen, p1_color,
                                       [(self.side_length + self.line_th) // 2 + self.side_length * x,
                                        (self.side_length + self.line_th) // 2 + self.side_length * y],
                                        self.piece_size)
                elif board[self.height - y - 1, x, 1] == 1:
                    pygame.draw.circle(self.screen, p2_color,
                                       [(self.side_length + self.line_th) // 2 + self.side_length * x,
                                        (self.side_length + self.line_th) // 2 + self.side_length * y],
                                       self.piece_size)
        
        """Render possible moves"""
        possible_moves = self.game.get_moves()
        for move in possible_moves:
            new_possible_color = (possible_color[0], possible_color[1], 200*self.weights[move])
            if self.tictactoe:
                move = (self.width * self.height) - (move)//self.width * self.width + (move) % self.width-self.width
            pygame.draw.circle(self.screen, new_possible_color, [
                    (self.side_length + self.line_th) // 2 + self.side_length * (
                                (self.Config.move_to_number(move)) % self.width),
                    (self.side_length + self.line_th) // 2 + self.side_length * (
                                (self.Config.move_to_number(move)) // self.width)], self.piece_size)
        pygame.display.flip()


    
    def _render_tictactoe(self):
        """Update screen for Tic tac toe"""
        background_color = (0, 109, 50)
        line_color = self.black
        p1_color = self.white
        p2_color = self.black
        possible_color = (255, 0, 0)
        self._render(background_color, line_color, p1_color, p2_color, possible_color)
        


    def _render_fourinarow(self):
        """Update screen for Four in a Row"""
        background_color = (0, 100, 150)
        line_color = self.black
        p1_color = (255, 0, 0)
        p2_color = (255, 255, 0)
        possible_color = (0, 255, 0)
        self._render(background_color, line_color, p1_color, p2_color, possible_color)
        
    def update_screen(self):
        """updates the screen with the right graphics"""
        if self.tictactoe:
            self._render_tictactoe()
        elif self.fourinarow:
            self._render_fourinarow()



    def execute_move(self):
        """Find the mouse position and execute move based on it"""
        if self.tictactoe:
            self.mouse_pos=(self.mouse_pos[0],self.height*self.side_length-self.mouse_pos[1])
        self.game.execute_move(self.Config.number_to_move((self.mouse_pos[1] - 2) // self.side_length * self.width + (self.mouse_pos[0] - 2) // self.side_length))#må generaliseres
        sleep(0.2) # Delay for preventing multiple presses accidently

