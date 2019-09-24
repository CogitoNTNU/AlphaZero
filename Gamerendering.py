from TicTacToe import Gamelogic
from TicTacToe import Config
import pygame
import sys


class TicTacToeRendering:

    def __init__(self, game):
        pygame.init()
        self.game = game
        self.side_length = 50
        self.line_th = 5
        self.height = 3
        self.width = 3
        #self.extrawidth = 150
        self.image = pygame.image.load("nevraltnett.png")
        self.imagerect = self.image.get_size()
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.move_color = (255, 0, 0)
        self.background_color=(0, 109, 50)
        self.piece_size = 20
        self.screen = pygame.display.set_mode([self.side_length * self.width + self.line_th + self.imagerect[0], max(self.side_length * self.height + self.line_th,self.imagerect[1])])
        while True:
            self.mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif pygame.mouse.get_pressed()[0] and self.mouse_pos[0] < self.side_length * self.width + self.line_th and self.mouse_pos[1] < self.side_length * self.height + self.line_th: #Sjekk om musen er innenfor brettet
                    self.execute_move()
                elif pygame.key.get_pressed()[32]:
                    self.game.undo_move()
                self._render_background()
                self._render_pieces()
                self._render_possible_moves()
                pygame.display.flip()

    def _render_background(self):
        self.screen.fill(self.background_color)
        for x in range(self.width + 1):
            pygame.draw.line(self.screen, self.black, [x * self.side_length + 2, 0], [x * self.side_length + 2, self.side_length * self.height + self.line_th-2],
                             self.line_th)
            pygame.draw.line(self.screen, self.black, [0, x * self.side_length + 2], [self.side_length * self.width + self.line_th-2, x * self.side_length + 2],
                             self.line_th)
        self.screen.blit(self.image,((self.side_length*self.width)+self.line_th,0))

    def _render_pieces(self):
        board = self.game.board
        for x in range(self.width):
            for y in range(self.height):
                if board[x, y, 1] == 1:
                    pygame.draw.circle(self.screen, self.white,
                                       [(self.side_length + self.line_th) // 2 + self.side_length * x, (self.side_length + self.line_th) // 2 + self.side_length * y], self.piece_size)
                elif board[x, y, 0] == 1:
                    pygame.draw.circle(self.screen, self.black, [(self.side_length + self.line_th) // 2 + self.side_length * x, (self.side_length + self.line_th) // 2 + self.side_length * y],
                                       self.piece_size)

    def _render_possible_moves(self):
        possible_moves = self.game.get_moves()
        for move in possible_moves:
            pygame.draw.circle(self.screen, self.move_color, [(self.side_length + self.line_th) // 2 + self.side_length * (Config.move_to_number(move)//3), (self.side_length + self.line_th) // 2 + self.side_length * (Config.move_to_number(move) % 3)], self.piece_size)

    def render(self):
        pass

    def execute_move(self):
        self.game.execute_move(Config.number_to_move((self.mouse_pos[0] - 2) // self.side_length * 3 + (self.mouse_pos[1] - 2) // self.side_length))#må generaliseres


rendering = TicTacToeRendering(Gamelogic.TicTacToe())
