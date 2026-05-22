import numpy as np
import os
import json
import random
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
parser.add_argument("--test_problems", type=int, default=100)

@dataclass
class DQNConfig:
    seed: int = 42
    batch_size: int = 256
    hidden_features: int = 256
    target_net_update: float = 0.005
    lr: float = 3e-4
    replay_capacity: int = 10000
    replay_terminal_ratio: float = 0.1
    num_episodes: int = 5000 
    save_interval: int = 100

    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999

class DQN():
    def __init__(self, env: BlockWorldEnv, config: DQNConfig, N: int):
        self.num_blocks = N
        self.env = env
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- TRÉNUJI NA: {self.device} ---")

        self.steps_done = 0
        self.episodes = 0
        self.epsilon = config.epsilon_start

        self.memory = ReplayBuffer(config.replay_capacity, config.replay_terminal_ratio)

        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self.state_dim = 2 * N * (N + 1)
        self.n_actions = N * N

        self.q_net = DQNNetwork(self.state_dim, self.n_actions, config.hidden_features).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.n_actions, config.hidden_features).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.config.lr)

    def train(self):
        for _ in range(self.config.num_episodes):
            raw_state, _ = self.env.reset()

            state = self.state_to_tensor(raw_state).unsqueeze(0).to(self.device)
            mask = self.create_legal_mask(raw_state[0].get_actions()).unsqueeze(0).to(self.device)

            for t in count():
                action = self._act_epsilon_greedy(raw_state)

                raw_next_state, reward, terminated, truncated, _ = self.env.step(action)

                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float)
                reward_tensor += torch.tensor(
                    self.compute_reward_shaping(raw_state, raw_next_state),
                    device=self.device, dtype=torch.float
                )

                if terminated:
                    next_state = None
                    next_mask = None
                else:
                    next_state = self.state_to_tensor(raw_next_state).unsqueeze(0).to(self.device)
                    next_mask = self.create_legal_mask(raw_next_state[0].get_actions()).unsqueeze(0).to(self.device)

                action_idx = self._action_to_index(action)
                action_tensor = torch.tensor([[action_idx]], device=self.device)

                self.memory.push(state, mask, action_tensor, next_state, next_mask, reward_tensor)

                state = next_state
                raw_state = raw_next_state
                mask = next_mask

                self.steps_done += 1
                if self.steps_done % 4 == 0:
                    self.optimize_model()

                if terminated or truncated or t >= 150:
                    self.episodes += 1
                    self.epsilon = max(self.config.epsilon_end,
                                       self.epsilon * self.config.epsilon_decay)

                    if self.episodes % 10 == 0:
                        print(f"Episode {self.episodes}, steps: {t+1}, epsilon: {self.epsilon:.3f}")

                    if self.config.save_interval > 0 and self.episodes % self.config.save_interval == 0:
                        save_model(self, self.episodes)

                    break

        print("Training complete")

    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return

        transitions = self.memory.sample(self.config.batch_size, self.rng)

        states  = torch.cat([t.state for t in transitions]).to(self.device)
        masks   = torch.cat([t.mask for t in transitions]).to(self.device)
        actions = torch.cat([t.action for t in transitions]).to(self.device)
        rewards = torch.cat([t.reward for t in transitions]).to(self.device)

        non_terminal_mask = torch.tensor(
            [t.next_state is not None for t in transitions],
            device=self.device,
            dtype=torch.bool
        )

        non_terminal_next_states = torch.cat(
            [t.next_state for t in transitions if t.next_state is not None]
        ).to(self.device)

        non_terminal_next_masks = torch.cat(
            [t.next_mask for t in transitions if t.next_mask is not None]
        ).to(self.device)

        q_values = self.q_net(states, masks).gather(1, actions)

        next_q = torch.zeros(self.config.batch_size, device=self.device)

        with torch.no_grad():
            if non_terminal_mask.any():
                next_vals = self.target_net(non_terminal_next_states, non_terminal_next_masks)
                next_vals[~non_terminal_next_masks] = -float('inf')
                next_q[non_terminal_mask] = next_vals.max(1).values

        target = rewards.flatten() + self.config.gamma * next_q

        loss = self.criterion(q_values.flatten(), target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.config.target_net_update * p.data + (1 - self.config.target_net_update) * tp.data)

    def _on_map(self, block_state):
        on = {}
        for stack in block_state.state:
            for i, blk in enumerate(stack):
                on[blk] = stack[i - 1] if i > 0 else 0
        return on

    def state_to_tensor(self, state) -> torch.Tensor:
        state_obj, goal_obj = state
        N = self.num_blocks

        def encode(block_state):
            on = self._on_map(block_state)
            vec = np.zeros(N * (N + 1), dtype=np.float32)
            for blk in range(1, N + 1):
                support = on.get(blk, 0)
                vec[(blk - 1) * (N + 1) + support] = 1.0
            return vec

        return torch.tensor(
            np.concatenate([encode(state_obj), encode(goal_obj)]),
            dtype=torch.float32
        )

    def _action_to_index(self, action):
        what, where = action
        row_offset = (what - 1) * self.num_blocks
        col_offset = where - int(where > what)
        return row_offset + col_offset

    def create_legal_mask(self, actions: Iterable) -> torch.Tensor:
        all_actions = np.zeros(self.n_actions)
        for (what, where) in actions:
            idx = self._action_to_index((what, where))
            all_actions[idx] = 1
        return torch.tensor(all_actions, dtype=torch.bool)


    def network_action_to_action(self, idx):
        what = (idx // self.num_blocks) + 1
        where = idx % self.num_blocks
        where += (where >= what)
        return (what, where)
    
    def compute_reward_shaping(self, state, next_state) -> float:
        state_obj, goal_obj = state
        next_obj = next_state[0]

        g_on = self._on_map(goal_obj)
        s_on = self._on_map(state_obj)
        sn_on = self._on_map(next_obj)

        before = sum(1 for b in g_on if s_on.get(b) == g_on[b])
        after = sum(1 for b in g_on if sn_on.get(b) == g_on[b])

        return float(after - before) - 0.1

    def _act_epsilon_greedy(self, state):
        actions = state[0].get_actions()
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.act(state)

    def act(self, state):
        state_tensor = self.state_to_tensor(state).unsqueeze(0).to(self.device)
        mask = self.create_legal_mask(state[0].get_actions()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.q_net(state_tensor, mask)
            q_vals[~mask] = -float('inf')
            best_idx = q_vals.argmax().item()

        return self.network_action_to_action(best_idx)

def save_model(model: DQN, episode: int, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(model.config), f, indent=2)
            
    torch.save(model.q_net.state_dict(), f"{save_dir}/q_net_ep{episode}.pt")
    torch.save(model.target_net.state_dict(), f"{save_dir}/target_net_ep{episode}.pt")
    print(f"Saved at episode {episode}")

def load_model(model_dir: str, episode: int, N: int):
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config_dict = json.load(f)
    config = DQNConfig(**config_dict)
    
    env = BlockWorldEnv(N)
    model = DQN(env, config, N)

    model.q_net.load_state_dict(torch.load(f"{model_dir}/q_net_ep{episode}.pt", map_location=model.device, weights_only=True))
    model.target_net.load_state_dict(torch.load(f"{model_dir}/target_net_ep{episode}.pt", map_location=model.device, weights_only=True))

    model.epsilon = 0.0
    print(f"Loaded episode {episode}")
    return model

if __name__ == "__main__":
    args = parser.parse_args()

    if args.restore_episode > 0:
        qlearning = load_model(args.model_restore_path, args.restore_episode, args.num_blocks)
    else:
        env = BlockWorldEnv(args.num_blocks)
        qlearning = DQN(env, DQNConfig(), args.num_blocks)

    if not args.no_train and args.restore_episode <= 0:
        qlearning.train()

    test_env = BlockWorldEnv(args.num_blocks)

    solved = 0
    steps_list = []

    for i in range(args.test_problems):
        s, _ = test_env.reset()
        print(f"\nProblem {i}: {s[0]} -> {s[1]}")

        for step in range(50):
            a = qlearning.act(s)
            s, _, done, _, _ = test_env.step(a)

            if done:
                solved += 1
                steps_list.append(step + 1)
                break

    avg = sum(steps_list) / len(steps_list) if steps_list else float("inf")
    print(f"\nSolved {solved}/{args.test_problems}, avg steps = {avg}")