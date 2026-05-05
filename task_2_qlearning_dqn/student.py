from blockworld import BlockWorldEnv
import random

class QLearning():
    # don't modify the methods' signatures!
    def __init__(self, env: BlockWorldEnv):
        self.env = env
        self.alpha = 0.3
        self.gamma = 0.95
        self.epsilon = 1
        self.Q_table = {}

    def _make_hashable(self, state, goal):
        return (str(state),str(goal))
    def train(self):
        # Use BlockWorldEnv to simulate the environment with reset() and step() methods.

        # s = self.env.reset()
        # s_, r, done = self.env.step(a)
        num_int = 20000
        for i in range(num_int):
            # 1. Catch the observation and the extra info junk
            obs, info = self.env.reset()

            # 2. Unpack the observation into your true state and goal!
            state, goal = obs
            
            done = False
            for _ in range (100):
                current_state_key = (state,goal)
                action_to_do = None
                #Pick an Action

                # if isnt in Q_table
                if current_state_key not in self.Q_table:
                    self.Q_table[current_state_key] = {action:0.0 for action in state.get_actions()}
                    action_to_do = random.choice( state.get_actions() )
                else:
                    random_num = random.random()
                    if random_num < self.epsilon:   
                        action_to_do = random.choice( state.get_actions() )
                    else:
                        room_score = self.Q_table[current_state_key]
                        action_to_do = max(room_score , key=room_score.get)

                #Do the action
                new_state_, reward, done, truncated, info= self.env.step(action_to_do)
                actual_new_state, next_goal= new_state_

                new_state_key = (actual_new_state, next_goal)
                
                #Calculating the Rumor
                if done:
                    self.Q_table[current_state_key][action_to_do] = reward + 10
                    break

                heighest_score = 0
                if new_state_key not in self.Q_table:
                    self.Q_table[new_state_key] = {action:0.0 for action in actual_new_state.get_actions()}
                else:
                    room_score = self.Q_table[new_state_key]
                    heighest_score = max(room_score.values())
                #Updating the Q_table
                
                old_score = self.Q_table[current_state_key][action_to_do]
                self.Q_table[current_state_key][action_to_do] = old_score + self.alpha * ((reward - 0.1) + self.gamma * heighest_score - old_score)

                #Move foward
                state = actual_new_state

            self.epsilon = self.epsilon * 0.9999

    def act(self, s):
        # 1. The test environment passes 's' as (state, goal)
        state, goal = s
        
        # 2. Convert to the string key format you used in training
        key = (state, goal)
        
        # 3. Check if we have this room in our Q-table
        if key in self.Q_table:
            # Pick the best action based on our training
            room_scores = self.Q_table[key]
            return max(room_scores, key=room_scores.get)
        else:
            # If we hit an unknown room, we have to guess
            return random.choice(state.get_actions())

if __name__ == '__main__':
    ##chat created training
    N = 4
    env = BlockWorldEnv(N)
    qlearning = QLearning(env)

    # 1. TRÉNINK (přesně 30s)
    print("Trénuji...")
    qlearning.train()

    # 2. EVALUACE (1000 problémů)
    test_env = BlockWorldEnv(N)
    test_problems = 1000
    solved = 0
    steps_list = []

    print(f"Testuji na {test_problems} problémech...")
    for _ in range(test_problems):
        # Reset prostředí
        obs_raw = test_env.reset()
        # Ošetření formátu resetu (aby to nespadlo)
        s = obs_raw[0] if isinstance(obs_raw, tuple) and not hasattr(obs_raw[0], 'get_actions') else obs_raw

        done = False
        for step in range(50):  # limit 50 kroků na úlohu
            a = qlearning.act(s)
            res = test_env.step(a)
            s_next, r, done = res[0], res[1], res[2]

            s = s_next
            if done:
                solved += 1
                steps_list.append(step + 1)
                break

    # 3. VÝPOČET BODŮ PODLE BRUTE
    p = solved
    e = sum(steps_list) / len(steps_list) if steps_list else 50

    # σ (sigma) - koeficient úspěšnosti
    sigma = max(0, min(1, (p - 350) / (950 - 350)))
    # κ (kappa) - koeficient rychlosti
    kappa = max(0, min(1, (20 - e) / (20 - 6)))

    final_points = round(7 * sigma * kappa, 1)  # Brute zaokrouhluje na celé body, ale 1 des. místo je lepší pro ladění

    print("-" * 30)
    print(f"Výsledky pro N={N}:")
    print(f"Vyřešeno (p): {p}/{test_problems} (σ = {sigma:.2f})")
    print(f"Průměr kroků (e): {e:.2f} (κ = {kappa:.2f})")
    print(f"PŘEDPOKLÁDANÉ BODY: {final_points} / 7")
    print("-" * 30)

