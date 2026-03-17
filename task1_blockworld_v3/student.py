from blockworld import BlockWorld
from queue import PriorityQueue

class BlockWorldHeuristic(BlockWorld):
	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)
		self.goal_map = None
	
	
	def create_goal_map(self,goal):
		goal_state = goal.get_state()
		self.goal_map = {}
		# block : [top block, under block, the error score]
		
		for stack in goal_state:
			# if stack is single block, there is nothing on top or bottom 
			if len(stack) == 1:
				self.goal_map[stack[0]] = [0,0,1]
			else:
				for index, each_block in enumerate(stack):
					key = []
					if index == 0: 
						key = [0,stack[index+1], index+1]
					elif index == len(stack) - 1:
						key = [stack[index-1], 0, index+1]
					else:
						key = [stack[index-1], stack[index+1], index+1]

					self.goal_map[each_block] = key

	def heuristic(self, goal):
		self_state = self.get_state()
		self_goal = goal.get_state()

		if self.goal_map == None:
			self.create_goal_map(goal)
		
		score = 0.0

		if self_state == self_goal:
			return 0
		
		for stack in self_state:
			# block : [top block, under block, the error score]
			if len(stack) == 1:
				key = self.goal_map[stack[0]]
				if key[1] != 0:
					score += 0.5
			else:
				for index, block in enumerate(stack):
					key = self.goal_map[block]

					#checking top block
					if index == 0:
						if key[1] != stack[index+1]:
							score += key[2]

					#checking last ground block
					elif index == len(stack) - 1:
						if key[0] != stack[index-1]:
							score += 1
						if key[1] != 0:
							score += 0.5 

					else:
						if key[0] != stack[index-1]:
							score += 1
						if key[1] != stack[index+1]:
							score += key[2]

		# ToDo. Implement the heuristic here.

		return score

from queue import PriorityQueue

class AStar():
    def search(self, start, goal):
        # 1. Inicializace nástrojů
        queue = PriorityQueue()
        counter = 0  # Tie-breaker (aby Python neporovnával objekty, když mají stejné skóre)
        
        # Slovníky pro sledování:
        # came_from si pamatuje: {kam_jsem_prisel: (odkud_jsem_prisel, jakou_akci)}
        came_from = {}
        # cost_so_far si pamatuje nejlepší (nejkratší) cestu od startu (g skóre)
        cost_so_far = {}
        
        # Abychom mohli stavy dávat jako klíče do slovníku, musí to být tuple
        # POZNÁMKA PRO TEBE: Tady v tom převodu si budeš muset případně 
        # dořešit to seřazení sloupců, jak jsme se bavili!
        start_tuple = tuple(tuple(stack) for stack in start.get_state())
        goal_tuple = tuple(tuple(stack) for stack in goal.get_state())
        
        # Nastavení počátečního stavu
        came_from[start_tuple] = None
        cost_so_far[start_tuple] = 0
        
        # Výpočet počáteční priority f = g + h (zde g=0)
        start_h = start.heuristic(goal)
        queue.put((start_h, counter, start))
        
        # 2. Hlavní cyklus (dokud máme co prozkoumávat)
        while not queue.empty():
            # Vytáhneme uzel s nejnižším skóre
            priorita, _, current_node = queue.get()
            
            # Převedeme aktuální stav na tuple, abychom se s ním mohli ptát do slovníků
            current_tuple = tuple(tuple(stack) for stack in current_node.get_state())
            
            # A) Jsme v cíli?
            if current_tuple == goal_tuple:
                # Našli jsme cíl! Zavoláme pomocnou funkci pro vypsání cesty
                return self.rekonstruuj_cestu(came_from, current_tuple)
            
            # B) Projdeme všechny sousedy (možné tahy)
            for action, neighbor in current_node.get_neighbors():
                neighbor_tuple = tuple(tuple(stack) for stack in neighbor.get_state())
                
                # Cena g do souseda je cena do aktuálního stavu + 1 krok
                new_cost = cost_so_far[current_tuple] + 1
                
                # Pokud jsme u souseda ještě nebyli, NEBO jsme našli kratší cestu:
                if neighbor_tuple not in cost_so_far or new_cost < cost_so_far[neighbor_tuple]:
                    # Uložíme si novou nejlepší cenu
                    cost_so_far[neighbor_tuple] = new_cost
                    
                    # Spočítáme nové f = g + h pro prioritní frontu
                    h = neighbor.heuristic(goal)
                    f = new_cost + h
                    
                    # Vložíme do fronty (counter zvýšíme, aby byl unikátní)
                    counter += 1
                    queue.put((f, counter, neighbor))
                    
                    # Zaznamenáme si do mapy, kudy jsme do souseda přišli
                    came_from[neighbor_tuple] = (current_tuple, action)
                    
        # Pokud se fronta vyprázdní a cíl jsme nenašli (nemělo by se u BlockWorldu stát)
        return None

    # Pomocná funkce, která vezme slovník a sestaví z něj ten seznam akcí
    def rekonstruuj_cestu(self, came_from, current_tuple):
        path = []
        
        # Cestujeme od cíle zpátky na start (start má v came_from uloženo None)
        while came_from[current_tuple] is not None:
            # Vytáhneme si předchozí stav a akci
            previous_tuple, action = came_from[current_tuple]
            
            # Akci přidáme do seznamu
            path.append(action)
            
            # "Skočíme" do předchozího stavu a cyklus se opakuje
            current_tuple = previous_tuple
            
        # Protože jsme šli od cíle ke startu, akce jsou pozpátku. Musíme je otočit.
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