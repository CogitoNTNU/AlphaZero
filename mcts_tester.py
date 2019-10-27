from MCTS import MCTS
from TicTacToe.Gamelogic import TicTacToe

game = TicTacToe()
game.execute_move(0)
start_state = game.get_state()
mcts_test_obj = MCTS(start_state)