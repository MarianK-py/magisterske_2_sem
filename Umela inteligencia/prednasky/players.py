import random
from games import TicTacToe, Gomoku
import math

################################### PLAYERS ###################################

class Player:
    def choose_move(self, game, state):
        raise NotImplementedError



class AskingPlayer(Player):
    def choose_move(self, game, state):
        # Asks user (human) which move to take. Useful for debug.
        actions = game.actions(state)
        print("Choose one of the following positions: {}".format(actions))
        game.display_state(state, True)
        return int(input('> '))



class RandomPlayer(Player):
    def choose_move(self, game, state):
        # Picks random move from list of possible ones.
        return random.choice(game.actions(state))
    
class Tree:
    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.visits = 1
        self.win_score = 0

class MCTSearchPlayer(Player):


    def choose_move(self, game, state):
        # FIXME implement this method to return the best move according to Monte Carlo Tree Search algorithm
        ### Examples:
        # my_player = game.player_at_turn(state)
        # opponent = game.other_player(my_player)
        # board = game.board_in_state(state)
        # if board[0][1] == my_player: ...
        # possible_actions = game.actions(state)
        # some_action = possible_actions[0]
        # new_state = game.state_after_move(current_state, some_action)
        # if game.is_terminal(some_state): ...
        # utility = game.utility(some_state, my_player)2
        return game.actions(state)[0]  # dummy return



################################ MAIN PROGRAM #################################

if __name__ == '__main__':
    ## Print all moves of the game? Useful for debugging, annoying if it`s already working.
    show_moves = False

    # Testing/comparing the performances. Uncomment/

    # TicTacToe().play([RandomPlayer(), AskingPlayer()], show_moves=show_moves)
    # TicTacToe().play([MCTSearchPlayer(), AskingPlayer()], show_moves=show_moves)
    # TicTacToe().play([MCTSearchPlayer(), RandomPlayer()], show_moves=show_moves)

    TicTacToe().play_n_games([MCTSearchPlayer(), RandomPlayer()], n=100)
    TicTacToe().play_n_games([MCTSearchPlayer(), MCTSearchPlayer()], n=100)
