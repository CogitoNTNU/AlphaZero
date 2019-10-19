# --Requires--:
# game.get_moves()
# game.execute_move()
# game.undo_move()
# game.is_final()
# game.get_score()
# game.get_states()
# get_board()
# get_turn()

import numpy as np
from FourInARow.Config import name, board_dims


class FourInARow:
    def __init__(self):

        """ A board of size (6, 7, 2) where the size is 7 along the horizontal axis and 6 along the vertical axis. """
        self.board = np.zeros((6, 7, 2), dtype=int)
        self.history = []
        self.name = name
        self.board_dims = board_dims

    def get_moves(self):
        """ Checks whether the uppermost space in the column is empty. """
        return [x for x in range(7) if self.board[-1, x, 0] == self.board[-1, x, 1] == 0]

    def get_legal_NN_output(self):
        """ A boolean array over the moves, with a 1 if the move represented by the index is legal. """
        return [1 if self.board[-1, x, 0] == self.board[-1, x, 1] == 0 else 0 for x in range(7)]

    def __find_uppermost_empty(self, column):
        """
        Returns the uppermost empty row in the column
        :param column: The column to check
        """
        for i in range(0, self.board.shape[0]):
            if self.board[i, column, 0] == self.board[i, column, 1] == 0:
                return i
        return self.board.shape[0]

    def execute_move(self, move):
        """ Places a piece in the column given by 'move' parameter if it is legal. """
        if move in self.get_moves():
            row = self.__find_uppermost_empty(move)
            self.board[row, move, len(self.history) % 2] = 1
            self.history.append(move)
            """ Registers a win if the last move results in a win. """
            self.__won()
        else:
            print('Illegal move')

        return self

    def undo_move(self):
        """ Undoes the last move. """
        if len(self.history) > 0 or self.history[-1] is None:
            move = self.history[-1]
            self.history.pop()
            self.board[self.__find_uppermost_empty(move) - 1, move, len(self.history) % 2] = 0
        else:
            print('could not undo move')

    def __valid_coordinates(self, row, column):
        """ Checks whether the given row and column is a valid coordinate on the board. """
        return 0 <= row < self.board.shape[0] and 0 <= column < self.board.shape[1]

    def __won(self):
        """ Returns true if the current board configuration is a winning configuration  """
        if len(self.history) == 0:
            return False
        """ The possible directions """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = 1 * (len(self.history) % 2 == 0)
        column = self.history[-1]
        columns = []
        if column is None:
            for i in range(self.board.shape[1]):
                if self.__find_uppermost_empty(i) > 0:
                    columns.append(i)
        else:
            columns.append(column)
        for column in columns:
            row = self.__find_uppermost_empty(column) - 1
            for dir in directions:
                num_in_row = 1

                """ Search in positive direction """
                r, c = row, column
                r += dir[0]
                c += dir[1]
                while self.__valid_coordinates(r, c) and self.board[r, c, player]:
                    num_in_row += 1
                    r += dir[0]
                    c += dir[1]

                """ Search in negative direction """
                r, c = row, column
                r -= dir[0]
                c -= dir[1]
                while self.__valid_coordinates(r, c) and self.board[r, c, player]:
                    num_in_row += 1
                    r -= dir[0]
                    c -= dir[1]
                if num_in_row >= 4:
                    return True
        return False

    def is_final(self):
        return self.__won() or len(self.history) == self.board.shape[0] * self.board.shape[1]

    def get_score(self):
        if self.is_final():
            if self.__won():
                return 2
            else:
                return 1
        else:
            print('not final')

    def get_outcome(self):
        if self.is_final():
            if self.__won():
                return [1, -1] if len(self.history) % 2 == 1 else [-1, 1]
            else:
                return [0, 0]

        else:
            print("not finished")

    def get_state(self):
        # return [str(self.get_board())]
        return str(self.history)

    def get_turn(self):
        return len(self.history) % 2 if not self.is_final() else None

    def get_board(self):
        return self.board if len(self.history) % 2 == 0 else np.flip(self.board, -1)

    def print_board(self):
        for y in range(0, self.board.shape[0])[::-1]:
            string = '|'
            for x in range(self.board.shape[1]):
                string += 'X' * int(self.board[y, x, 0] == 1)
                string += 'O' * int(self.board[y, x, 1] == 1)
                string += ' ' * int(self.board[y, x, 0] == self.board[y, x, 1] == 0)
                string += '|'
            print(string)
    
    def player_turn(self):
        p1 = 0
        p2 = 0
        for row in self.board:
            for rute in row:
                if rute[0] == 1:
                    p1 += 1
                elif rute[1] == 1:
                    p2 += 1
        if p1 == p2:
            return 0
        else:
            return 1


    def get_moves_from_board_state(self,board):
        game = self.create_game(board)
        return game.get_moves()

    @classmethod
    def create_game(cls, state):
        game = cls()
        game.board = state
        game.history = np.sum(state)*[None]
        return game

# game = FourInARow()
# print(game.board.shape)
# game.print_board()
# print(game.get_moves())
# print(game.get_score())
# print(len(game.history))
# while True:
#     inp = int(input("Number:"))
#     game.execute_move(inp)
#     game.print_board()
#     if game.is_final():
#         print("Player " + str(1-(len(game.history) % 2)) + " won.")
#     #game.undo_move()