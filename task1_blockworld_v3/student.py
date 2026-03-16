from blockworld import BlockWorld

class BlockWorldHeuristic(BlockWorld):
	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)
		self.goal_map = None
	
	
	def create_goal_map(self):
		goal_state = goal.get_state()
		self.goal_map = {}

		for stack in goal_state:
			for idx, block in enumerate(stack):
				if idx < len(stack) - 1:
					next_block = stack[idx+1]
					self.goal_map[block] = [next_block,idx+1] 			#height - how many move we need to change oder to get what we want
				else:
					self.goal_map[block] = [0, idx+1]

	def heuristic(self, goal):
		self_state = self.get_state()
		goal_state = goal.get_state()

		# ToDo. Implement the heuristic here.

		return 0.

class AStar():
	def search(self, start, goal):
		# ToDo. Return a list of optimal actions that takes start to goal.
		
		# You can access all actions and neighbors like this:
		# for action, neighbor in state.get_neighbors():
		# 	...

		return None

if __name__ == '__main__':
	# Here you can test your algorithm. You can try different N values, e.g. 6, 7.
	N = 5

	start = BlockWorldHeuristic(N)
	goal = BlockWorldHeuristic(N)

	print("Searching for a path:")
	print(f"{start} -> {goal}")
	print()

	astar = AStar()
	path = astar.search(start, goal)

	if path is not None:
		print("Found a path:")
		print(path)

		print("\nHere's how it goes:")

		s = start.clone()
		print(s)

		for a in path:
			s.apply(a)
			print(s)

	else:
		print("No path exists.")

	print("Total expanded nodes:", BlockWorld.expanded)