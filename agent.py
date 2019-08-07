import numpy as np
from numpy import random
from ConvAutoencoder import Conv_Autoencoder

class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """
        self.iteration_counter = 0
        # Variables related to random exploration
        self.random_straight_range = 20
        self.random_turn_range = 20
        self.random_straight = random.randint(1, self.random_straight_range)
        self.random_turn = random.randint(1, self.random_turn_range)
        self.random_action = 0
        self.random_movement_counter = 0
        # self.vision = Conv_Autoencoder()
        self.video_frames = []
        self.video_dump_count = 0

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """

    def step(self, obs=0, reward=0, done=0, info=0):
        """
        A single step the agent should take based on the current state of the environment
        We will run the Gym environment (AnimalAIEnv) and pass the arguments returned by env.step() to
        the agent.

        Note that should if you prefer using the BrainInfo object that is usually returned by the Unity
        environment, it can be accessed from info['brain_info'].

        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """

        self.iteration_counter += 1
        self.random_movement_counter += 1
        self.video_frames.append(obs)
        if len(self.video_frames) == 10000:
            video_data = np.concatenate(self.video_frames)
            np.save("video_data/normed_video_frames_"+str(self.video_dump_count), video_data)
            self.video_dump_count+=1
            self.video_frames = []

        return

    def random_exploration(self):
        """Generate a series of straight moves followed by a series of turning moves in a random direction"""
        # check if series of random actions has been completed and if so reset variables
        if self.random_movement_counter > self.random_turn + self.random_straight:
            # reset variables
            self.random_movement_counter = 0
            self.random_action = random.randint(0, 2)
            self.random_straight = random.randint(1, self.random_straight_range+1)
            self.random_turn = random.randint(1, self.random_turn_range+1)

        # take a series of straight movements
        if self.random_movement_counter <= self.random_straight:
            return [1, 0]

        # take a series of rotation movements
        elif self.random_straight < self.random_movement_counter <= self.random_turn + self.random_straight:
            if self.random_action == 0:
                return [0, 1]
            else:
                return [0, 2]