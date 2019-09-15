# --Requires--:
# game.get_moves()
# game.execute_move()
# game.undo_move()
# game.is_final()
# game.get_score()
# game.get_states()
# get_board()
# get_turn()

# TODO: update find moves

import numpy as np
import time


class TicTacToe:
    def __init__(self):
        self.board = np.array([[[0, 0], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [0, 0]]])
        self.history = []

    def get_moves(self):
        return [x for x in range(9) if self.board[x // 3, x % 3, 0] == self.board[x // 3, x % 3, 1] == 0]

    def get_legal_NN_output(self):
        return [1 if self.board[x // 3, x % 3, 0] == self.board[x // 3, x % 3, 1] == 0 else 0 for x in range(9)]
        # moves = []
        # for x in range(9):
        #     if self.board[x // 3, x % 3, 0] == self.board[x // 3, x % 3, 1] == 0:
        #         moves.append(x)
        # return moves

    def execute_move(self, move):
        self.board[move // 3, move % 3, len(self.history) % 2] = 1
        self.history.append(move)
        # poss_moves = self.get_moves()
        # if move in poss_moves:
        #     self.board[move // 3, move % 3, len(self.history) % 2] = 1
        #     self.history.append(move)
        # else:
        #     print('illegal move')

    def undo_move(self):
        if len(self.history) > 0:
            move = self.history[-1]
            self.board[move // 3, move % 3, (len(self.history) - 1) % 2] = 0
            self.history.pop()
        else:
            print('could not undo move')

    def _won(self):
        player = 1 * (len(self.history) % 2 == 0)
        for x in range(3):
            # Horizontal
            if self.board[x, 0, player] == self.board[x, 1, player] == self.board[x, 2, player] != 0:
                return True
            # Vertical
            if self.board[0, x, player] == self.board[1, x, player] == self.board[2, x, player] != 0:
                return True
        # Diagonal
        if self.board[0, 0, player] == self.board[1, 1, player] == self.board[2, 2, player] != 0:
            return True
        if self.board[0, 2, player] == self.board[1, 1, player] == self.board[2, 0, player] != 0:
            return True

    def is_final(self):
        if self._won():
            return True
        if len(self.history) == 9:
            return True
        return False

    def get_score(self):
        if self.is_final():
            if self._won():
                return 2
            else:
                return 1
        else:
            print('not final')

    def get_outcome(self):
        if self.is_final():
            if self._won():
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
        for x in range(3):
            string = '|'
            for y in range(3):
                string += 'X' * int(self.board[x, y, 0] == 1)
                string += 'O' * int(self.board[x, y, 1] == 1)
                string += ' ' * int(self.board[x, y, 0] == self.board[x, y, 1] == 0)
                string += '|'
            print(string)

# game = TicTacToe()
# game.print_board()
# while True:
#     inp = int(input("Number:"))
#     game.execute_move(inp)
#     game.print_board()
#     game.undo_move()
