import numpy as np
import os
import json
import random
import ast
import re

# BLINDFOLD PYTORCH: Tell it no GPUs exist on this machine to stop the driver warning
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from dataclasses import dataclass
from argparse import ArgumentParser
from itertools import count
from typing import Iterable, Tuple

from blockworld import BlockWorldEnv
from dqn_networks import DQNNetwork
from replay_buffer import ReplayBuffer

parser = ArgumentParser()
parser.add_argument("--num_blocks", type=int, default=5)
parser.add_argument("--model_restore_path", type=str, default="checkpoints")
parser.add_argument("--restore_episode", type=int, default=-1)
parser.add_argument("--no_train", action='store_true')
parser.add_argument("--test_problems", type=int, default=1000)

@dataclass
class DQNConfig:
    seed: int = 42
    batch_size: int = 128
    hidden_features: int = 256
    target_net_update: float = 0.005
    lr: float = 3e-4
    replay_capacity: int = 10000
    replay_terminal_ratio: float = 0.1
    num_episodes: int = 1000
    save_interval: int = 100
    # DQN specific parameters
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500000

class DQN():
    def __init__(self, env: BlockWorldEnv, config: DQNConfig, N: int):
        self.num_blocks = N
        self.env = env
        self.config = config
        self.device = torch.device('cpu')
        
        self.steps_done = 0
        self.episodes = 0
        self.memory = ReplayBuffer(config.replay_capacity, config.replay_terminal_ratio)
        
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        
        # State dimension: N for current state + N for goal state
        self.state_dim = self.num_blocks * 2 
        self.n_actions = N * N

        self.q_net = DQNNetwork(self.state_dim, self.n_actions, config.hidden_features).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.n_actions, config.hidden_features).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.config.lr)

    def train(self):
        for i_episode in range(self.config.num_episodes):
            raw_state, info = self.env.reset()
            state = self.state_to_tensor(raw_state).unsqueeze(0).to(self.device)
            mask = self.create_legal_mask(raw_state[0].get_actions()).unsqueeze(0).to(self.device)

            for t in range(200):
                # Epsilon-greedy action selection
                epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                          np.exp(-1. * self.steps_done / self.config.epsilon_decay)
                self.steps_done += 1

                if random.random() > epsilon:
                    with torch.no_grad():
                        q_values = self.q_net(state, mask)
                        q_values[~mask] = -float('inf') 
                        action_idx = q_values.max(1)[1].item()
                        action = self.network_action_to_action(action_idx)
                else:
                    action = random.choice(raw_state[0].get_actions())

                # Convert action back to index for storing in buffer
                what, where = action
                row = what - 1
                col = where - int(where > what)
                action_idx = row * self.num_blocks + col
                action_tensor = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)

                raw_next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float)
                reward_shaping = torch.tensor(self.compute_reward_shaping(raw_state, raw_next_state), device=self.device, dtype=torch.float)
                reward_tensor += reward_shaping

                if terminated:
                    next_state = None
                    next_mask = None
                else:
                    next_state = self.state_to_tensor(raw_next_state).unsqueeze(0).to(self.device)
                    next_mask = self.create_legal_mask(raw_next_state[0].get_actions()).unsqueeze(0).to(self.device)

                self.memory.push(state, mask, action_tensor, next_state, next_mask, reward_tensor)

                state = next_state
                raw_state = raw_next_state
                mask = next_mask

                self.optimize_model()

                # Soft update of the target network
                target_net_state_dict = self.target_net.state_dict()
                q_net_state_dict = self.q_net.state_dict()
                for key in q_net_state_dict:
                    target_net_state_dict[key] = q_net_state_dict[key]*self.config.target_net_update + target_net_state_dict[key]*(1-self.config.target_net_update)
                self.target_net.load_state_dict(target_net_state_dict)

                if terminated or truncated or t == 199:
                    self.episodes += 1
                    if self.episodes % 10 == 0:
                        print(f"Finished episode {self.episodes}. Performed {t + 1} steps. Epsilon: {epsilon:.2f}")
                    if self.config.save_interval > 0 and self.episodes % self.config.save_interval == 0:
                        save_model(self, self.episodes)
                    break

        print('Complete')

    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return

        transitions = self.memory.sample(self.config.batch_size, self.rng)
        
        state_batch = torch.cat([t[0] for t in transitions])
        mask_batch = torch.cat([t[1] for t in transitions])
        action_batch = torch.cat([t[2] for t in transitions])
        reward_batch = torch.cat([t[5] for t in transitions])
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s[3] is not None, transitions)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s[3] for s in transitions if s[3] is not None])
        non_final_next_action_masks = torch.cat([s[4] for s in transitions if s[4] is not None])

        state_action_values = self.q_net(state_batch, mask_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.config.batch_size, device=self.device)
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                next_q_values = self.target_net(non_final_next_states, non_final_next_action_masks)
                next_q_values[~non_final_next_action_masks] = -float('inf')
                next_state_values[non_final_mask] = next_q_values.max(1)[0]
                
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

    def state_to_tensor(self, state) -> torch.Tensor:
        current_blocks, goal_blocks = state
        
        curr_str = re.sub(r'np\.int64\((\d+)\)', r'\1', str(current_blocks))
        goal_str = re.sub(r'np\.int64\((\d+)\)', r'\1', str(goal_blocks))
        
        current_list = ast.literal_eval(curr_str)
        goal_list = ast.literal_eval(goal_str)

        state_tensor = torch.zeros(self.state_dim, dtype=torch.float32)
        
        for stack in current_list:
            bottom = 0 
            for block in stack:
                state_tensor[block - 1] = bottom
                bottom = block 

        for stack in goal_list:
            bottom = 0 
            for block in stack:
                state_tensor[self.num_blocks + block - 1] = bottom
                bottom = block

        return state_tensor

    def create_legal_mask(self, actions: Iterable) -> torch.Tensor:
        all_actions = np.zeros(self.n_actions)

        for (what, where) in actions:
            row_offset = (what - 1) * self.num_blocks
            col_offset = where - int(where > what)
            all_actions[row_offset + col_offset] = 1
        return torch.tensor(all_actions, dtype=torch.bool)

    def compute_reward_shaping(self, state, next_state) -> float:
        current_tensor = self.state_to_tensor(state)
        next_tensor = self.state_to_tensor(next_state)
        
        curr_correct = torch.sum(current_tensor[:self.num_blocks] == current_tensor[self.num_blocks:]).item()
        next_correct = torch.sum(next_tensor[:self.num_blocks] == next_tensor[self.num_blocks:]).item()
        
        if next_correct > curr_correct:
            return 0.5
        elif next_correct < curr_correct:
            return -0.5
        return -0.01

    def network_action_to_action(self, network_action: np.ndarray) -> Tuple[int, int]:
        network_action = int(network_action)
        what = (network_action // self.num_blocks) + 1
        where = network_action % self.num_blocks
        where += (where >= what)
        return (what, where)

    def act(self, state):
        # We removed the raw_state unpacking because Brute hands us the perfect tuple!
        state_tensor = self.state_to_tensor(state).unsqueeze(0).to(self.device)
        
        # state[0] holds the current blocks, which is what get_actions() needs
        mask = self.create_legal_mask(state[0].get_actions()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor, mask)
            q_values[~mask] = -float('inf') 
            action_idx = q_values.max(1)[1].item()
            
        return self.network_action_to_action(action_idx)

def load_model(model_dir: str, N: int) -> 'DQN':
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config_dict = json.load(f)
    config = DQNConfig(**config_dict)
    env = BlockWorldEnv(N)
    model = DQN(env, config, N)
    model.q_net.load_state_dict(torch.load(os.path.join(model_dir, "q_net.pt"), weights_only=True, map_location=model.device))
    model.target_net.load_state_dict(torch.load(os.path.join(model_dir, "target_net.pt"), weights_only=True, map_location=model.device))
    print(f"Loaded weights from '{model_dir}/'")
    return model

def save_model(model: DQN, episode: int, save_dir: str = "checkpoints") -> None:
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, f"config.json")
    env_path = os.path.join(save_dir, f"env_spec.pt")
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(model.config), f, indent=2)
    if not os.path.exists(env_path):
        torch.save(model.num_blocks, env_path)
    torch.save((model.rng.bit_generator.state, torch.get_rng_state()), os.path.join(save_dir, f"rng_states_ep{episode}.pt"))
    torch.save(model.q_net.state_dict(), os.path.join(save_dir, f"q_net_ep{episode}.pt"))
    torch.save(model.target_net.state_dict(), os.path.join(save_dir, f"target_net_ep{episode}.pt"))
    print(f"Saved weights at episode {episode} to '{save_dir}/'")

def nn_learner_setup(N: int, model_dir: str = None):
    env = BlockWorldEnv(N)
    if model_dir:
        learner = load_model(model_dir, N)
    else:
        config = DQNConfig()
        learner = DQN(env, config, N)
    return learner

if __name__ == '__main__':
    args = parser.parse_args()
    if args.restore_episode > 0 and args.model_restore_path:
        qlearning = nn_learner_setup(args.num_blocks, args.model_restore_path)
    else:
        qlearning = nn_learner_setup(args.num_blocks)

    if not args.no_train:
        qlearning.train()

    test_env = BlockWorldEnv(args.num_blocks)
    test_problems = args.test_problems
    solved = 0
    avg_steps = []

    for test_id in range(test_problems):
        s, _ = test_env.reset()
        done = False
        print(f"\nProblem {test_id}:")
        print(f"{s[0]} -> {s[1]}")

        for step in range(50):
            a = qlearning.act(s)
            print(f"{a}: {s[0]}")
            s_, r, done, truncated, _ = test_env.step(a)
            s = s_

            if done:
                solved += 1
                avg_steps.append(step + 1)
                break

    avg_steps = sum(avg_steps) / len(avg_steps) if avg_steps else float('inf')
    print(f"Solved {solved}/{test_problems} problems, with average number of steps {avg_steps}.")