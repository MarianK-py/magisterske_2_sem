import pyddl
import sys
import simulator

# Vypracoval: Marian Kravec

class world:
    __map = {}
    __totalGold = 0
    __maxX = 0
    __maxY = 0
    __startX = 0
    __startY = 0

    def load(self, file):
        f = open(file, 'r')
        data = f.read().split('\n')
        f.close()
        self.__parse(data)

    def __parse(self, data):
        y = 0
        for line in data:
            self.__maxY = y
            x = 0
            for char in line:
                self.__maxX = max(self.__maxX, x)
                self.__map[(x,y)] = char
                if char == 'g':
                    self.__totalGold+=1
                if char == '@':
                    self.__startX = x
                    self.__startY = y
                x+=1
            y+=1

    def getProblem(self):
        init = [
            ('=', ('carrying_arrow', 'player'), 0),
            ('=', ('carrying_gold', 'player'), 0)
        ]

        for i in range(self.__maxX):
            for j in range(self.__maxY):
                if self.__map[(i,j)] == 'g':
                    init.append(("gold", i, j))
                    init.append(("free", i, j))
                elif self.__map[(i,j)] == '@':
                    init.append(("at", "player", i, j))
                    init.append(("free", i, j))
                elif self.__map[(i,j)] == ' ':
                    init.append(("free", i, j))
                elif self.__map[(i,j)] == 'W':
                    init.append(("wumpus", i, j))
                elif self.__map[(i,j)] == 'A':
                    init.append(("arrow", i, j))
                    init.append(("free", i, j))
                elif self.__map[(i,j)] == '#':
                    init.append(("wall", i, j))

        goal = [
            ('=', ('carrying_gold', 'player'), self.__totalGold),
            #('=', ('carrying_gold', 'player'), 2),
            ('at', 'player', self.__startX, self.__startY)
        ]

        positions = tuple(range(0, max([self.__maxY, self.__maxX])+1))

        for i in positions:
            init.append(("left", i, i-1))
            init.append(("right", i, i+1))
            init.append(("up", i, i-1))
            init.append(("down", i, i+1))

        actions = [
            pyddl.Action('move-left',
                         parameters = (('position', 'px'),
                                      ('position', 'py'),
                                      ('position', 'bx')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('left', 'px', 'bx'),
                                          ('free', 'bx', 'py')),
                         effects = (pyddl.neg(('at', 'player', 'px', 'py')),
                                    ('at', 'player', 'bx', 'py'))),
            pyddl.Action('move-right',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'bx')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('right', 'px', 'bx'),
                                          ('free', 'bx', 'py')),
                         effects = (pyddl.neg(('at', 'player', 'px', 'py')),
                                    ('at', 'player', 'bx', 'py'))),
            pyddl.Action('move-up',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'by')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('up', 'py', 'by'),
                                          ('free', 'px', 'by')),
                         effects = (pyddl.neg(('at', 'player', 'px', 'py')),
                                    ('at', 'player', 'px', 'by'))),
            pyddl.Action('move-down',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'by')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('down', 'py', 'by'),
                                          ('free', 'px', 'by')),
                         effects = (pyddl.neg(('at', 'player', 'px', 'py')),
                                    ('at', 'player', 'px', 'by'))),

            pyddl.Action('take-gold',
                         parameters = (('position', 'px'),
                                       ('position', 'py')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('gold', 'px', 'py')),
                         effects = (pyddl.neg(('gold', 'px', 'py')),
                                    ('+=', ('carrying_gold', 'player'), 1))),

            pyddl.Action('take-arrow',
                         parameters = (('position', 'px'),
                                       ('position', 'py')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('arrow', 'px', 'py')),
                         effects = (pyddl.neg(('arrow', 'px', 'py')),
                                    ('+=', ('carrying_arrow', 'player'), 1))),

            pyddl.Action('shoot-wumpus-left',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'bx')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('left', 'px', 'bx'),
                                          ('wumpus', 'bx', 'py'),
                                          ('>', ('carrying_arrow', 'player'), 0)),
                         effects = (pyddl.neg(('wumpus', 'bx', 'py')),
                                    ('free', 'bx', 'py'),
                                    ('-=', ('carrying_arrow', 'player'), 1))),
            pyddl.Action('shoot-wumpus-right',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'bx')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('right', 'px', 'bx'),
                                          ('wumpus', 'bx', 'py'),
                                          ('>', ('carrying_arrow', 'player'), 0)),
                         effects = (pyddl.neg(('wumpus', 'bx', 'py')),
                                    ('free', 'bx', 'py'),
                                    ('-=', ('carrying_arrow', 'player'), 1))),
            pyddl.Action('shoot-wumpus-up',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'by')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('up', 'py', 'by'),
                                          ('wumpus', 'px', 'by'),
                                          ('>', ('carrying_arrow', 'player'), 0)),
                         effects = (pyddl.neg(('wumpus', 'px', 'by')),
                                    ('free', 'px', 'by'),
                                    ('-=', ('carrying_arrow', 'player'), 1))),
            pyddl.Action('shoot-wumpus-down',
                         parameters = (('position', 'px'),
                                       ('position', 'py'),
                                       ('position', 'by')),
                         preconditions = (('at', 'player', 'px', 'py'),
                                          ('down', 'py', 'by'),
                                          ('wumpus', 'px', 'by'),
                                          ('>', ('carrying_arrow', 'player'), 0)),
                         effects = (pyddl.neg(('wumpus', 'px', 'by')),
                                    ('free', 'px', 'by'),
                                    ('-=', ('carrying_arrow', 'player'), 1))),
        ]

        domain = pyddl.Domain(actions)

        problem = pyddl.Problem(
            domain,
            {
                'position': tuple(positions),
            },
            init=tuple(init),
            goal=tuple(goal),
        )
        return problem

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Map file need to be specified!")
        print("Example: python3 " + sys.argv[0] + " world1.txt")
        sys.exit(1)
    w = world()
    w.load(sys.argv[1])
    problem = w.getProblem()
    plan = pyddl.planner(problem, verbose=True)
    if plan is None:
        print('Hunter is not able to solve this world!')
    else:
        actions = [action.name for action in plan]
        print(", ".join(actions))
        f = open(sys.argv[1] + ".solution", "w")
        f.write("\n".join(actions))
        f.close()
        input()
        simulator.simulate(sys.argv[1], sys.argv[1] + ".solution")

#%%
