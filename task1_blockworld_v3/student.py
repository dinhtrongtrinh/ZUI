import numpy as np
np.int = int
from blockworld import BlockWorld
import heapq 

class BlockWorldHeuristic(BlockWorld):
    def __init__(self, num_blocks=5, state=None):
        BlockWorld.__init__(self, num_blocks, state)

    def heuristic(self, goal):
        self_state = self.get_state()
        goal_state = goal.get_state()
        def get_support_dict(state_list):
            support = {}
            for stack in state_list:
                bottom = 0
                for block in reversed(stack):
                    support[block] = bottom
                    bottom = block
            return support
            
        self_supp = get_support_dict(self_state)
        goal_supp = get_support_dict(goal_state)
        
        h = 0
        for stack in self_state:
            is_correct_so_far = True
            seen_below = set()
            for block in reversed(stack):
                if is_correct_so_far:
                    supp = self_supp[block]
                    if goal_supp.get(block) != supp:
                        is_correct_so_far = False
                if not is_correct_so_far:
                    h += 1
                    target = goal_supp.get(block, 0)
                    if target != 0 and target in seen_below:
                        h += 1
                seen_below.add(block)
                
        return float(h)

class AStar():
    def get_state_tuple(self, node):
    
        return tuple(sorted(tuple(stack) for stack in node.get_state()))

    def search(self, start, goal):
        queue = []
        counter = 0
        
        start_tuple = self.get_state_tuple(start)
        goal_tuple = self.get_state_tuple(goal)
        
        came_from = {start_tuple: None}
        cost_so_far = {start_tuple: 0}
        
        start_h = start.heuristic(goal)
        
        heapq.heappush(queue, (start_h, counter, start, start_tuple))
        
        while queue:
            _, _, current_node, current_tuple = heapq.heappop(queue)
            
            if current_tuple == goal_tuple:
                return self.reconstruct_path(came_from, current_tuple)
            
            current_cost = cost_so_far[current_tuple]
            
            for action, neighbor in current_node.get_neighbors():
                neighbor_tuple = self.get_state_tuple(neighbor)
                new_cost = current_cost + 1
                
                if neighbor_tuple not in cost_so_far or new_cost < cost_so_far[neighbor_tuple]:
                    cost_so_far[neighbor_tuple] = new_cost
                    h = neighbor.heuristic(goal)
                    f = new_cost + h
                    
                    counter += 1
                    heapq.heappush(queue, (f, counter, neighbor, neighbor_tuple))
                    came_from[neighbor_tuple] = (current_tuple, action)
                    
        return None

    def reconstruct_path(self, came_from, current_tuple):
        path = []
        while came_from[current_tuple] is not None:
            previous_tuple, action = came_from[current_tuple]
            path.append(action)
            current_tuple = previous_tuple
            
        path.reverse()
        return path

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