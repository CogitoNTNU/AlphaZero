from Othello import Gamelogic
import pygame
import sys


class OthelloRendering:

    def __init__(self, game):
        pygame.init()
        self.game = game
        self.side_length = 50
        self.line_th = 5
        self.screen = pygame.display.set_mode(
            [self.side_length * 8 + self.line_th, self.side_length * 8 + self.line_th])
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif pygame.mouse.get_pressed()[0]:
                    self.execute_move()
                elif pygame.key.get_pressed()[32]:
                    self.game.undo_move()
                self._render_background()
                self._render_pieces()
                self._render_possible_moves()
                pygame.display.flip()

    def _render_background(self):
        self.screen.fill([0, 109, 50])
        for x in range(9):
            pygame.draw.line(self.screen, (0, 0, 0), [x * self.side_length + 2, 0], [x * self.side_length + 2, 403],
                             self.line_th)
            pygame.draw.line(self.screen, (0, 0, 0), [0, x * self.side_length + 2], [403, x * self.side_length + 2],
                             self.line_th)

    def _render_pieces(self):
        board = self.game.get_board()
        for x in range(8):
            for y in range(8):
                if board[x, y, 1] == 1:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       [28 + self.side_length * x, 28 + self.side_length * y], 20)
                elif board[x, y, 0] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0), [28 + self.side_length * x, 28 + self.side_length * y],
                                       20)

    def _render_possible_moves(self):
        possible_moves = self.game.get_moves()
        for move in possible_moves:
            pygame.draw.circle(self.screen, (255, 0, 0), [28 + 50 * move[0], 28 + 50 * move[1]], 20)

    def render(self):
        pass

    def execute_move(self):
        pos = pygame.mouse.get_pos()
        self.game.execute_move([(pos[0] - 2) // 50, (pos[1] - 2) // 50])


rendering = OthelloRendering(Gamelogic.Othello())
