from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
from agent import Agent
import sys
from numpy import random

env = UnityEnvironment(
    file_name='env/AnimalAI',  # Path to the environment
    worker_id=random.randint(1,10),  # Unique ID for running the environment (used for connection)
    seed=0,  # The random seed
    docker_training=False,  # Whether or not you are training inside a docker

    n_arenas=1,  # Number of arenas in your environment
    play=False,  # Set to False for training
    inference=True,  # Set to true to watch your agent in action
    resolution=None  # Int: resolution of the agent's square camera (in [4,512], default 84)
)

if len(sys.argv) > 1:
    arena_config_in = ArenaConfig(sys.argv[1])
else:
    arena_config_in = ArenaConfig('examples/configs/allObjectsRandom.yaml')

env.reset(arenas_configurations=arena_config_in,
          # A new ArenaConfig to use for reset, leave empty to use the last one provided
          train_mode=True  # True for training
          )
# arenas_configurations="examples/configs/1-Food.yaml",
# no_graphics=False,  # Always set to False
agent = Agent()

while True:
    take_action_vector = agent.random_exploration()
    info_dict = env.step(vector_action=take_action_vector)
    brain_info = info_dict["Learner"]
    visual_observation = brain_info.visual_observations[0]
    agent.step(visual_observation)


