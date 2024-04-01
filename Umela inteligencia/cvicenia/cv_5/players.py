import random
from games import TicTacToe, Gomoku
import math

# Vypracoval: Marian Kravec

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
    C = 1
    sims = 0

    def monte_carlo_tree_search(self, root, total_sims, game, player):
        while self.sims < total_sims:
            leaf = self.traverse(game, root)
            simulation_result = self.rollout(game, leaf.state, player)
            self.backpropagate(leaf, simulation_result)
            self.sims += 1

        bestNewState = self.best_UCB(root).state

        bestAction = self.get_action(game, root.state, bestNewState)

        return bestAction

    def get_action(self, game, old_state, new_state):
        pos_act = game.actions(old_state)
        for act in pos_act:
            state = game.state_after_move(old_state, act)
            if state == new_state:
                return act
        return None

    def fully_expanded(self, game, node):
        return len(node.children) == len(game.actions(node.state))

    def get_child_states(self, node):
        s = []
        for i in node.children:
            s.append(i.state)
        return s

    def pick_unvisited(self, game, node):
        act = game.actions(node.state)
        states = self.get_child_states(node)
        for i in act:
            ns = game.state_after_move(node.state, i)
            t = Tree(ns, node)
            if ns not in states:
                node.children.append(t)
                return t
        return None

    def UCB(self, node):
        return (node.win_score/node.visits) + self.C*math.sqrt(math.log(self.sims)/(node.visits))

    def best_UCB(self, node):
        return max(node.children, key=lambda x: self.UCB(x))

    def traverse(self, game, node):
        node.visits += 1
        while self.fully_expanded(game, node) and not game.is_terminal(node.state):
            node = self.best_UCB(node)
            node.visits += 1

        return self.pick_unvisited(game, node) or node

    def rollout(self, game, state, player):
        while not game.is_terminal(state):
            state = game.state_after_move(state, random.choice(game.actions(state)))
        return game.utility(state, player)

    def backpropagate(self, node, result):
        if node.parent == None:
            return
        node.win_score += result
        self.backpropagate(node.parent, result)

    def choose_move(self, game, state):
        self.sims = 0

        my_player = game.player_at_turn(state)
        #opponent = game.other_player(my_player)
        root = Tree(state)

        act = self.monte_carlo_tree_search(root, 300, game, my_player)

        return act #game.actions(state)[0]



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
