import simulator
import sys
import math
from enum import Enum
import random

#Marian Kravec

gamma = 0.95 # gamma in (0, 1), update this value as needed

# possible actions
class Actions(Enum):
	Up = 1
	Left = 2
	Down = 3
	Right = 4

class world:
	__map = {} # world map, each (x,y) cell also represents one state
	__golds = {} # values of golds
	__steps = 0 # maximum allowed steps
	__moveR = 0 # movement reward
	__deathR = 0 # death reward
	__maxX = 0 # size of world x axis
	__maxY = 0 # size of world y axis
	__startX = 0 # treasure hunter start x coordinate
	__startY = 0 # treasure hunter start y coordinate
	__policy = {} # action selection policy, a map from state (position x,y) to action (Actions class)
	__policy_vals = {}

	def load(self, file):
		f = open(file, 'r')
		data = f.read().split('\n')
		f.close()
		self.__parse(data)
		
	# map data parser, do not touch this!
	# it will fill the variables of the world class
	def __parse(self, data):
		self.__moveR = float(data[0])
		self.__deathR = float(data[1])
		self.__steps = int(data[2])
		golds = int(data[3])
		for i in range(4, 4 + golds):
			posVal = data[i].split(' ')
			self.__golds[(int(posVal[0]), int(posVal[1]))] = float(posVal[2])
		startFrom = 4 + golds
		y = 0
		for i in range(startFrom, len(data)):
			x = 0
			for c in data[i]:
				if c == 'H':
					self.__startX = x
					self.__startY = y
					self.__map[(x,y)] = ' '
				else:
					self.__map[(x,y)] = c
				x+=1
			self.__maxX = max(self.__maxX, x)
			y+=1
		self.__maxY = y
		
	def trainPolicy(self):
		change = True
		new_policy = {}
		new_policy_vals = {}
		while change:
			change = False
			for j in range(self.__maxX):
				for k in range(self.__maxY):
					pos = (j, k)
					new_val_act = max([
						(self.__newPolicy(pos, Actions.Up), Actions.Up),
						(self.__newPolicy(pos, Actions.Down), Actions.Down),
						(self.__newPolicy(pos, Actions.Left), Actions.Left),
						(self.__newPolicy(pos, Actions.Right), Actions.Right)
					], key=(lambda x: x[0]))
					if new_val_act[1] != self.__policy.get(pos, Actions.Up):
						change = True
					new_policy[pos] = new_val_act[1]
					new_policy_vals[pos] = new_val_act[0]
			self.__policy = new_policy
			self.__policy_vals = new_policy_vals

		pass # implement value iteration algorithm and establish final policy

	# function to compute new policy for pos and action
   	def __newPolicy(self, pos, action):
		good_pos = self.__newPosition(pos, action)
		bad_poss = self.__getBadMovesForAction(pos, action)
		new_pol = 0
		new_pol += 0.8*(self.__getStateChangeReward(good_pos) + gamma*self.__policy_vals.get(good_pos, 0))
		new_pol += 0.1*(self.__getStateChangeReward(bad_poss[0]) + gamma*self.__policy_vals.get(bad_poss[0], 0))
		new_pol += 0.1*(self.__getStateChangeReward(bad_poss[1]) + gamma*self.__policy_vals.get(bad_poss[1], 0))
		return new_pol

	# returns new position according to the direction of choosen action, may return the same position if the target is a wall
	def __newPosition(self, pos, action):
		if action == Actions.Up:
			newPos = (pos[0], pos[1] - 1)
		elif action == Actions.Down:
			newPos = (pos[0], pos[1] + 1)
		elif action == Actions.Left:
			newPos = (pos[0] - 1, pos[1])
		else:
			newPos = (pos[0] + 1, pos[1])
		
		if newPos not in self.__map or self.__map[newPos] == '#':
			return pos
		else:
			return newPos
			
	# returns list of two position which can be reached in case of failure of the selected action
	def __getBadMovesForAction(self, pos, action):
		if action == Actions.Up:
			return [self.__newPosition(pos, Actions.Left), self.__newPosition(pos, Actions.Right)]
		elif action == Actions.Down:
			return [self.__newPosition(pos, Actions.Left), self.__newPosition(pos, Actions.Right)]
		elif action == Actions.Right:
			return [self.__newPosition(pos, Actions.Up), self.__newPosition(pos, Actions.Down)]
		else:
			return [self.__newPosition(pos, Actions.Up), self.__newPosition(pos, Actions.Down)]
	
	# returns reward for moving the selected position
	def __getStateChangeReward(self, posTo):
		if self.__map[posTo] == ' ':
			return self.__moveR
		elif self.__map[posTo] == 'G':
			if posTo in self.__golds:
				return self.__golds[posTo]
			else:
				return self.__moveR
		elif self.__map[posTo] == 'O':
			return self.__deathR
		else:
			return 0
			
		
	def findTreasures(self):
		pos = (self.__startX, self.__startY)
		steps = [pos]
		
		stepsCount = 0
		
		while stepsCount < self.__steps:
			stepsCount+=1
			
			action = self.__getPolicyAction(pos)
			pos = self.__invokeAction(pos, action)
			steps.append(pos)
			
			if pos in self.__map and (self.__map[pos] == 'O' or self.__map[pos] == 'G'):
				break
		
		return steps
		
	def __getPolicyAction(self, pos):
		if pos in self.__policy:
			return self.__policy[pos]
		else:
			return Actions.Up
		
	# performs action invokation from current position, will return new position
	def __invokeAction(self, pos, action):
		goodPos = self.__newPosition(pos, action)
		badPositions = self.__getBadMovesForAction(pos, action)
		
		p = random.uniform(0, 1)
		
		if p < 0.8:
			return goodPos
		elif p >= 0.8 and p < 0.9:
			return badPositions[0]
		else:
			return badPositions[1]
	
if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Map file need to be specified!")
		print("Example: python3 " + sys.argv[0] + " world1.txt")
		sys.exit(1)
	w = world()
	w.load(sys.argv[1])
	#w.load("mapa1.txt")
	w.trainPolicy()
	steps = w.findTreasures()
	simulator.simulate(sys.argv[1], steps)
	#simulator.simulate("mapa1.txt", steps)
	