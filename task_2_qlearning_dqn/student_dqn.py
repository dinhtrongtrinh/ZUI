
import numpy as np
import os
import json
import random
import torch

from dataclasses import dataclass
from argparse import ArgumentParser
from itertools import count
from typing import Iterable,Tuple


from blockworld import BlockWorldEnv
from dqn_networks import DQNNetwork
from replay_buffer import ReplayBuffer


#Arguments to control which experiment will be run. Do not modify these!
parser = ArgumentParser()
parser.add_argument("--num_blocks", type=int, default=5, help="How many blocks to allow in our BlockWorld instance.")
parser.add_argument("--model_restore_path", type=str, default="checkpoints", help="Path from where to restore a stored model. Used only for DQN")
parser.add_argument("--restore_episode", type=int, default=-1, help="An episode of the stored model to restore. If <=0, the model is trained from scratch. Used only for DQN")
parser.add_argument("--no_train", action='store_true', help="If this flag is supplied, no training is performed and the model is run as is.")
parser.add_argument("--test_problems", type=int, default=1000, help="How many problems to test the learner on.")


@dataclass
class DQNConfig:
    #Do not delete these! But you can play with the values.
    #PRNG seed, to get consistent results.
    seed: int = 42
    #Minibatch size
    batch_size: int = 128
    #Network parameters
    hidden_features: int = 256
    #Update rate of the target network
    target_net_update: float = 0.005
    #Overall learning rate
    lr: float = 3e-4
    #Replay buffer parameters
    replay_capacity: int = 10000
    #A prioritized replay trick. Since our
    # environment normally has sparse rewards (only positive reward for correct solution).
    # it is quite hard task for the network to learn. We maintain a separate buffer
    # for the terminal steps, which actually give us our reward.
    # This ratio defines the fraction of the size of the
    # terminal-only buffer w.r.t. the normal buffer.
    # For details, look at the ReplayBuffer implementation.
    replay_terminal_ratio: float = 0.1
    #How many environment episodes to run
    num_episodes: int = 1000
    #How often to checkpoint the model. It will be saved at the interval of episodes defined by this.
    save_interval: int = 100
		#TODO: Your other parameters go here.

class DQN():
	# don't modify the methods' signatures!
  def __init__(self, env: BlockWorldEnv, config: DQNConfig, N: int):
    """Class handling the Deep Q-learning algorithm, for better scaling than
    tabular version in BlockWorld.

    Args:
        env (BlockWorldEnv): The BlockWorldEnvironment
        config (DQNConfig): Configuration containing all parameters required by DQN
        N (int): Number of blocks in the blockworld. We need this to determine action space size.
    """
    self.num_blocks = N
    self.env = env
    self.config = config
    #If interested you can try training on GPU too, but it requires a compatible torch installation.
    self.device = torch.device('cpu')
    self.config = config
    self.env = env

    self.steps_done = 0
    self.episodes = 0

    self.memory = ReplayBuffer(config.replay_capacity, config.replay_terminal_ratio)

    #Fix the random seed, for reproducibility
    self.rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    #For the neural network, we require a fixed
    # size input/output.
    # TODO: Fill in the correct dimension here
    # based on your chosen state representation
    self.state_dim = 1
    #For actions, we can move every element at any other element (except itself)
    # and ground. That means N actions for each element, however not all
    # of them will always be legal. We will have to employ legal action masking
    self.n_actions = N * N

    self.q_net = DQNNetwork(self.state_dim, self.n_actions, config.hidden_features)
    self.target_net = DQNNetwork(self.state_dim, self.n_actions, config.hidden_features)
    self.target_net.load_state_dict(self.q_net.state_dict())

    # A standard Mean squared error loss.
    self.criterion = torch.nn.MSELoss()

    #Recommended to use the Adam optimizer.
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.config.lr)

  def train(self):
    for i_episode in range(self.config.num_episodes):
      raw_state, info = self.env.reset()

      #This is how to pre-compute tensors and add a batch dimension of 1
      state = self.state_to_tensor(raw_state).unsqueeze(0)
      #Our legal action mask
      mask = self.create_legal_mask(raw_state[0].get_actions()).unsqueeze(0)

      #Effectivelly the same as while True with maintaining a counter.

      #TODO: Fill in this skeleton training loop with the actual
			# DQN training loop and network optimization.
      for t in count():
        action = self.act(raw_state)
        raw_next_state, reward, terminated, truncated, _ = self.env.step(action)
        #This is how to convert to a torch tensor.
        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float)
        #Add our reward shaping term
        reward_shaping = torch.tensor(self.compute_reward_shaping(raw_state, raw_next_state), device=self.device, dtype=torch.float)
        reward_tensor += reward_shaping
        if terminated:
            next_state = None
            next_mask = None
        else:
            # Pre-compute next state tensors
            next_state = self.state_to_tensor(raw_next_state).unsqueeze(0)
            next_mask = self.create_legal_mask(raw_next_state[0].get_actions()).unsqueeze(0)
        state = next_state
        raw_state = raw_next_state
        mask = next_mask
        if terminated:
            self.episodes += 1
            print(f"Finished episode {self.episodes}. Performed {t + 1} steps.")
            if self.config.save_interval > 0 and self.episodes % self.config.save_interval == 0:
                save_model(self, self.episodes)
            break

    print('Complete')

  def state_to_tensor(self, state) ->torch.Tensor:
    """Convert a state of the BlockWorld environment into a flat tensor.

    Args:
        state (_type_): The state tuple provided by the BlockWorld

    Returns:
        torch.Tensor: A torch suitable flat representation of the state.
    """
    #TODO: Create some suitable representation of the state
    state_tensor = torch.zeros(self.state_dim)
    return state_tensor

  def create_legal_mask(self, actions: Iterable) ->torch.Tensor:
    """Create a mask of legal actions in the given state,
    to be represented as just a binary vector, like the
    network expects it.

    Args:
        actions (Iterable): An iterable of (where, what) actions
        as returned by the environment

    Returns:
        torch.Tensor: The binary vector as torch Tensor.
    """
    all_actions = np.zeros(self.n_actions)
    for (what, where) in actions:
      row_offset = (what - 1) * self.num_blocks
      #Compensate for the fact, that we
      # cannot move a block upon itself.
      # So, blocks with a higher
      # number than our block will have index one higher
      # in the action map than we want it to
      col_offset = where - int(where > what)
      all_actions[row_offset + col_offset] = 1
    return torch.tensor(all_actions, dtype=torch.bool)

  def compute_reward_shaping(self, state, next_state) -> float:
    """Computes the reward shaping when moving from state to next_state
    This serves to help the DQN drive the search to promising space,
    as just the standard spare reward of -0.1 or 10 makes the task much harder.

    Args:
        state: The current environment state
        next_state: The environment state we transitioned to.

    Returns:
        float: The reward shaping component, that will be added up to the original reward.
    """
	#TODO: Come up with a suitable reward shaping, that will drive
	# the learning towards the goal.
    return 0

  def network_action_to_action(self, network_action: np.ndarray) ->Tuple[int, int]:
    """Convert action index returned by the network Q-values back to the
    environment (what, where) tuple

    Args:
        network_action (torch.Tensor| np.ndarray): Index of the chosen action

    Returns:
        Tuple[int, int]: A (what, where) tuple, as provided by the environment
    """
    network_action = int(network_action)
    #Blocks are 1 indexed, so add 1
    what = (network_action // self.num_blocks) + 1
    where = network_action % self.num_blocks
    #Moving a block upon itself is illegal,
    # so if the where index is >= our what index
    # we must add 1 to get the correct block
    where += (where >= what)
    return (what, where)

  def act(self, state):
    """Return an action to play in a given state"""
    random_action = random.choice( state[0].get_actions() )
    return random_action



def load_model(model_dir: str, N: int) -> 'DQN':
		"""Restore a model from a directory containing config.json, q_net.pt and target_net.pt."""
		# Load config from JSON (safe, no pickle)
		with open(os.path.join(model_dir, "config.json"), 'r') as f:
			config_dict = json.load(f)
		config = DQNConfig(**config_dict)

		env = BlockWorldEnv(N)
		model = DQN(env, config, N)

		model.q_net.load_state_dict(
				torch.load(os.path.join(model_dir, "q_net.pt"), weights_only=True,
										map_location=model.device))
		model.target_net.load_state_dict(
				torch.load(os.path.join(model_dir, "target_net.pt"), weights_only=True,
										map_location=model.device))
		print(f"Loaded weights from '{model_dir}/'")
		return model

def save_model(model: DQN, episode: int, save_dir: str = "checkpoints") -> None:
		"""Save policy and target network weights to disk."""
		os.makedirs(save_dir, exist_ok=True)
		#Save the static information, config and environment.
		# Save only once, if it does not exist yet
		config_path = os.path.join(save_dir, f"config.json")
		env_path = os.path.join(save_dir, f"env_spec.pt")
		if not os.path.exists(config_path):
				# Save config as JSON (safe format, no pickle needed)
				with open(config_path, 'w') as f:
					json.dump(vars(model.config), f, indent=2)
		if not os.path.exists(env_path):
				torch.save(model.num_blocks, env_path)
		#Save the rng states
		torch.save((model.rng.bit_generator.state, torch.get_rng_state()), os.path.join(save_dir, f"rng_states_ep{episode}.pt"))
		#Save the replay buffer state, optional.
		#torch.save(model.memory.getstate(), os.path.join(save_dir, f"replay_buffer_ep{episode}.pt"))
		#Save the weights
		torch.save(model.q_net.state_dict(),
								os.path.join(save_dir, f"q_net_ep{episode}.pt"))
		torch.save(model.target_net.state_dict(),
								os.path.join(save_dir, f"target_net_ep{episode}.pt"))
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

	# Train
	if not args.no_train:
		qlearning.train()

	# Evaluate
	test_env = BlockWorldEnv(args.num_blocks)

	test_problems = args.test_problems
	solved = 0
	avg_steps = []

	for test_id in range(test_problems):
		s, _ = test_env.reset()
		done = False

		print(f"\nProblem {test_id}:")
		print(f"{s[0]} -> {s[1]}")

		for step in range(50): 	# max 50 steps per problem
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
