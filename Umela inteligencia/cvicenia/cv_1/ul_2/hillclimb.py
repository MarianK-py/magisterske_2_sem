import random

class HillClimb:
    def __init__(self, position, n_gold, golds):
        self.position = position
        self.n_gold = n_gold
        self.g_positions = golds
        self.n_steps = 10
        self.grid = 10
        self.directions = ['U', 'D', 'L', 'R']

    def eval_fitness(self, route):
        # We simulate movement on grid and if we land
        # on cell with gold we take gold and remove
        # that cell from list of positions containing gold
        reward = 0
        poss_gold = self.g_positions.copy()
        found_gold = []
        pos = self.position.copy()
        for dir in route:
            pos = self.move(pos, dir)
            if pos in poss_gold:
                reward += 1
                found_gold.append(pos)
                poss_gold.remove(pos)

        return reward

    def neighbours(self, route):
        nb = []
        # To get neighbours we use pretty straigh-forward approach
        # where we "mutate" (change) one step to different direction
        # and we compute all such routes that are different from original
        # in exactly one step
        for i, dir in enumerate(route):
            for ndir in self.directions:
                if dir != ndir:
                    new_route = route.copy()
                    new_route[i] = ndir
                    nb.append(new_route)

        return nb

    def search(self, n_attempts):
        att_val = []
        for att in range(n_attempts):
            # In each attempt we randomly generate route
            # and then compute all it's neighbours and found
            # one with best fitness
            # if fitness increase we continue computing neighbours
            # of new best route
            # if fitness decrease we end computation because
            # no neighbour is better then current best route
            # if fitness don't change we still look at neighbours of neighbour
            # as is it as good as previous solution but might have better neighbour
            # if even after three repetitions fitness don't increase we end computation
            best_route = random.choices(self.directions, k=10)
            old_max = -1
            new_max = self.eval_fitness(best_route)
            been_stuck = 0

            while new_max >= old_max and been_stuck < 3:
                old_max = new_max
                new_best_route = max(self.neighbours(best_route), key=lambda x: self.eval_fitness(x))
                new_max = self.eval_fitness(new_best_route)
                if new_max < old_max:
                    new_max = old_max
                    new_best_route = best_route
                    break
                elif new_max == old_max:
                    been_stuck += 1
                    best_route = new_best_route
                else:
                    been_stuck = 0
                    best_route = new_best_route
                    old_max = new_max
            att_val.append((new_max, new_best_route))
        print(max(att_val))
        return max(att_val)

    def move(self, pos, dir):
        # computation of new position after step
        if dir == 'U':
            pos[1] = (pos[1]-1)%self.grid
        elif dir == 'D':
            pos[1] = (pos[1]+1)%self.grid
        elif dir == 'L':
            pos[0] = (pos[0]-1)%self.grid
        elif dir == 'R':
            pos[0] = (pos[0]+1)%self.grid
        return pos


if __name__ == "__main__":
    f = open("data.txt", "r")

    data = f.readlines()
    data = [d.strip().split() for d in data]
    data = [[int(n) for n in d] for d in data]

    posit, num_g = data[0], data[1][0]
    g_positions = data[2:]

    HC = HillClimb(posit, num_g, g_positions)
    HC.search(10)
