import pygame
import sys
from time import sleep
import copy
import pydot
import heapq
import os
import random

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# from MCTS import MCTS
from Main import *

"""Weird bug when trying to import MCTS, so had to star import from Main"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

class GameRendering:

    def __init__(self, game, agent, Config, numSearch):
        """Initialize the pygame"""
        pygame.init()
        pygame.font.init()
        self.default_font = pygame.font.get_default_font()
        self.text_size = 10
        self.font_color = (40, 40, 40)
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
        self.image = pygame.image.load("Images/nevraltnett.png")
        self.imagerect = (0, 0)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.piece_size = self.side_length // 3
        self.screen = pygame.display.set_mode([self.side_length * self.width + self.line_th + self.imagerect[0],
                                               max(self.side_length * (self.height + 1) + self.line_th,
                                                   self.imagerect[1])])  # + self.imagerect[0],self.imagerect[1]
        self.weights = [0] * len(self.game.get_moves())  # Set weight to empty array to represent all moves
        self.count = 0  # Switches sides of human and machine
        self.primary_line = []
        self.won = 0
        self.tied = 0
        self.lost = 0

        """find what game is given"""
        self.tictactoe = False
        self.fourinarow = False
        if self.game.name == "TicTacToe":
            self.tictactoe = True
            self.background_color = (0, 109, 50)
        elif self.game.name == "FourInARow":
            self.fourinarow = True
            self.background_color = (0, 100, 150)
        """check if graphviz is installed and in path"""
        try:
            self.test_graph = "digraph g{rankdir=LR;testing -> testing -> tested}"
            self.test_graph = pydot.graph_from_dot_data(self.test_graph)[0]
            self.test_graph.write_png('Images/graph.png')
        except FileNotFoundError:
            print("Error:Graphviz is not installed or not on path, skipping visualization")
            self.draw_graph = False
        except:
            print("Error: Unknown error with graph visualization, skipping")
            self.draw_graph = False
        else:
            self.draw_graph = True

        self.update_screen()

        """continuosly check for updates"""
        while True:
            self.mouse_pos = pygame.mouse.get_pos()
            if self.game.is_final():
                sleep(4)
                """Show death screen"""
                self.screen.fill(self.black)
                font_size = max((self.side_length * self.width + self.line_th) // 35, 20)
                myfont = pygame.font.SysFont(self.default_font, 2 * font_size)
                """Who won"""
                if self.game.get_outcome()[0] == 0:
                    winner = myfont.render('Tied', False, (255, 255, 255))
                    self.tied += 1
                elif (self.game.get_outcome()[0] == 1 and self.count % 2 == 0) or (
                        self.game.get_outcome()[1] == 1 and self.count % 2 == 1):
                    winner = myfont.render('AI won!', False, (255, 255, 255))
                    self.lost += 1
                else:
                    winner = myfont.render('Human won', False, (255, 255, 255))
                    self.won += 1
                self.screen.blit(winner, (
                (self.side_length * self.width + self.line_th + self.imagerect[0]) // 2 - winner.get_width() / 2,
                self.side_length // 3 - winner.get_height() // 2))

                myfont = pygame.font.SysFont('Comic Sans MS', font_size)
                switch_side = myfont.render('(Switching sides)', False, (0, 255, 0))
                self.screen.blit(switch_side, (
                (self.side_length * self.width + self.line_th + self.imagerect[0]) // 2 - switch_side.get_width() / 2,
                self.side_length // 3 - switch_side.get_height() // 2 + winner.get_height()))

                """Shows the score"""
                myfont = pygame.font.SysFont(self.default_font, 2 * font_size)
                wtl = myfont.render('Win/Tie/Loss', False, self.white)
                self.screen.blit(wtl, (
                (self.side_length * self.width + self.line_th + self.imagerect[0]) // 2 - wtl.get_width() / 2,
                (self.side_length * self.height) // 2 - wtl.get_height() // 2))
                score = myfont.render(str(self.won) + "-" + str(self.tied) + "-" + str(self.lost), False, self.white)
                self.screen.blit(score, (
                (self.side_length * self.width + self.line_th + self.imagerect[0]) // 2 - score.get_width() / 2,
                (self.side_length * self.height) // 2 + wtl.get_height()))

                pygame.display.flip()
                print("GAME IS OVER")
                self.count += 1  # Switches sides

                sleep(1)  # Catch glitchy graohics
                sleep(4)  # Hold the death screen open
                """clean the board and graphics"""
                self.weights = [0] * len(self.weights)
                self.game.board = np.copy(self.start_pos)  # don't want to change it
                self.game.history = []
                self.imagerect = (0, 0)
                self.screen = pygame.display.set_mode(
                    [self.side_length * self.width + self.line_th + self.imagerect[0],
                     max(self.side_length * (self.height + 1) + self.line_th, self.imagerect[1])])
                self.update_screen()

            elif (self.game.get_turn() + self.count) % 2 and not self.game.is_final():
                """look for human input"""
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or pygame.key.get_pressed()[27]:  # Escape quits the game
                        sys.exit()
                    elif pygame.mouse.get_pressed()[0] and self.mouse_pos[
                        0] < self.side_length * self.width + self.line_th and self.mouse_pos[
                        1] < self.side_length * self.height + self.line_th:  # Check if mouse is pressed in board
                        self.execute_move()
                        self.update_screen()
                    elif pygame.key.get_pressed()[8]:  # Backspace to undo move
                        self.game.undo_move()
                        self.game.undo_move()
                        self.update_screen()

            elif not self.game.is_final():
                """If machines turn, machine do move"""
                tree = MCTS.MCTS(self.game, self.game.board, self.agent, self.Config)
                if len(self.game.get_moves()) > 1:  # Does not compute last possible move very deeply
                    for searches in range(numSearch):
                        tree.search()
                        if (searches % 100 == 0 and searches != 0):
                            """update weight on screen every 200 search"""
                            self.NNvisual(tree, num_nodes=20)

                    self.NNvisual(tree, num_nodes=20)
                else:
                    tree.search_series(numSearch)
                predict = tree.get_most_searched_move(tree.root)
                #                print("Stillingen vurderes som: ",self.agent.predict(np.array([self.game.get_board()]))[1])
                self.game.execute_move(predict)
                self.update_screen()
                self.show_gamelines(self.primary_line)
                self.see_valuation()

    def see_valuation(self):
        """see how the nn values different moves on its turn for itself"""
        if self.tictactoe:
            """Has to flip the logic in y direction since it increases down"""
            possible_moves = self.game.get_moves()
            for move in possible_moves:
                self.label = self.font_renderer.render(str(round(self.weights[move], 4)), 1, self.font_color)
                self.screen.blit(self.label, [(self.side_length + self.line_th) // 2 + self.side_length * (
                            (self.Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,
                                              self.side_length * ((8 - self.Config.move_to_number(
                                                  move)) // self.width) + self.label.get_height() - self.line_th])
                pygame.display.flip()
        elif self.fourinarow:
            possible_moves = self.game.get_moves()
            for move in possible_moves:
                self.label = self.font_renderer.render(str(round(self.weights[move], 4)), 1, self.font_color)
                self.screen.blit(self.label, [(self.side_length + self.line_th) // 2 + self.side_length * (
                            (self.Config.move_to_number(move)) % self.width) - self.label.get_width() / 2,
                                              (self.side_length // self.height) - self.label.get_height()])
                pygame.display.flip()

    def _render(self, background_color, line_color, p1_color, p2_color, possible_color):
        """Generic board maker"""

        """Background"""
        self.screen.fill(background_color)
        self.screen.blit(self.image, (self.width * self.side_length + self.line_th, 0))
        """Draw board lines
        pygame.draw.line(surface, color, start_pos, end_pos, width)"""
        for line in range(self.width + 1):
            """Vertical lines"""
            pygame.draw.line(self.screen, line_color,
                             [line * self.side_length + 2, 0],
                             [line * self.side_length + 2, self.side_length * self.height + self.line_th - 2],
                             self.line_th)
            """Horizontal lines"""
            if line <= self.height:
                pygame.draw.line(self.screen, line_color,
                                 [0, line * self.side_length + 2],
                                 [self.side_length * self.width + self.line_th - 2, line * self.side_length + 2],
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
            new_possible_color = (possible_color[0], possible_color[1], 200 * self.weights[move])
            if self.tictactoe:
                move = self.y_flip(move)
            pygame.draw.circle(self.screen, new_possible_color, [
                (self.side_length + self.line_th) // 2 + self.side_length * (
                        (self.Config.move_to_number(move)) % self.width),
                (self.side_length + self.line_th) // 2 + self.side_length * (
                        (self.Config.move_to_number(move)) // self.width)], self.piece_size)
            myfont = pygame.font.SysFont(self.default_font, self.piece_size)
            action = myfont.render(str(move), False, background_color)
            self.screen.blit(action, ((self.side_length + self.line_th) // 2 + self.side_length * (
                    (self.Config.move_to_number(move)) % self.width) - action.get_width() // 2,
                                      (self.side_length + self.line_th) // 2 + self.side_length * (
                                              (self.Config.move_to_number(
                                                  move)) // self.width) - action.get_height() // 2))
        pygame.display.flip()

    def _render_tictactoe(self):
        """Update screen for Tic tac toe"""
        line_color = self.black
        p1_color = self.white
        p2_color = self.black
        possible_color = (255, 0, 0)
        self._render(self.background_color, line_color, p1_color, p2_color, possible_color)

    def _render_fourinarow(self):
        """Update screen for Four in a Row"""
        line_color = self.black
        p1_color = (255, 0, 0)
        p2_color = (255, 255, 0)
        possible_color = (0, 255, 0)
        self._render(self.background_color, line_color, p1_color, p2_color, possible_color)

    def update_screen(self):
        """updates the screen with the right graphics"""
        if self.tictactoe:
            self._render_tictactoe()
        elif self.fourinarow:
            self._render_fourinarow()

    def execute_move(self):
        """Find the mouse position and execute move based on it"""
        if self.tictactoe:
            self.mouse_pos = (self.mouse_pos[0], self.height * self.side_length - self.mouse_pos[1])
        self.game.execute_move(self.Config.number_to_move((self.mouse_pos[1] - 2) // self.side_length * self.width + (
                    self.mouse_pos[0] - 2) // self.side_length))  # må generaliseres
        sleep(0.2)  # Delay for preventing multiple presses accidently

    def build_graph(self, graph_root, tree_root, graph, heap):
        heapq.heappush(heap, (100000 - tree_root.get_times_visited(), random.randint(1, 10000000000), tree_root))
        # graph.add_node(node)
        for child in tree_root.children:
            self.build_graph(None, child, graph, heap)
        # if graph_root:
        #     graph.add_edge(pydot.Edge(graph_root, node, label=str("a")))

    def visualize_tree(self, root, num_nodes=20):
        heap = []
        graph = pydot.Dot(graph_type='graph')
        self.build_graph(None, root, graph, heap)
        for x in range(num_nodes):
            if heap == []:
                break
            top_heap = heapq.heappop(heap)
            tree_node, node_visits = top_heap[2], top_heap[0]
            node = pydot.Node(id(tree_node), style='filled',
                              fillcolor="#aa88aa",
                              label=str(- node_visits + 100000), shape="circle", fixedsize="shape")
            graph.add_node(node)
            if x != 0:
                move = tree_node.get_last_action()
                if self.tictactoe:
                    move = self.y_flip(move)
                graph.add_edge(pydot.Edge(id(tree_node.parent), id(tree_node), label=str(move)))
        """roterer grafen, setter bakgrunn"""
        graph_string = graph.to_string()
        graph_string = graph_string.replace("{", '{rankdir = LR;bgcolor="#%02x%02x%02x";' % self.background_color)
        graph = pydot.graph_from_dot_data(graph_string)[0]
        graph.write_png('graph.png')

    def NNvisual(self, tree, num_nodes):
        '''visualize tree'''
        if self.draw_graph:
            self.visualize_tree(tree.root, num_nodes)
            self.image = pygame.image.load("graph.png")
            self.imagerect = self.image.get_size()
            self.image = pygame.transform.smoothscale(self.image,
                                                      (min(720, self.imagerect[0]), min(720, self.imagerect[1])))
            self.imagerect = self.image.get_size()
            self.screen = pygame.display.set_mode(
                [self.side_length * self.width + self.line_th + self.imagerect[0],
                 max(self.side_length * (self.height + 1) + self.line_th, self.imagerect[1])])
        self.weights = tree.get_posterior_probabilities()
        """Build most searched line, and show it on screen"""
        self.primary_line = []
        best_action = tree.get_most_searched_child_node(tree.root)
        while best_action != None:
            if self.tictactoe:
                self.primary_line.append(self.y_flip(best_action.last_action))
            else:
                self.primary_line.append(best_action.last_action)
            best_action = tree.get_most_searched_child_node(best_action)
        '''update screen'''
        self.update_screen()
        self.see_valuation()
        self.show_gamelines(self.primary_line)

    def show_gamelines(self, pline):
        myfont = pygame.font.SysFont(self.default_font, 30)
        line = myfont.render('Projected line:' + str(pline), False, self.white)
        self.screen.blit(line, (0, (self.side_length * self.height) + self.line_th))
        pygame.display.flip()

    def y_flip(self, move):
        # for use with TicTacToe
        return (self.width * self.height) - (move) // self.width * self.width + (move) % self.width - self.width