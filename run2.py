# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:17:16 2020

@author: seano
"""

from mlagents_envs.environment import UnityEnvironment
from numpy.random import choice
import h5py
import matplotlib.pyplot as plt
# =============================================================================
# from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
# =============================================================================

NUM_SAMPLES = 100000
H5FILE= "data/simulation_data_1608_100000steps.h5"
with h5py.File(H5FILE, 'w') as f:
       vec_data = f.create_dataset('vector_obs', (NUM_SAMPLES, 5))
       vis_data = f.create_dataset('visual_obs', (NUM_SAMPLES, 256, 256, 3))
print("File ready fi dem")
env_name = "C:/Users/seano/Desktop/unity_builds/craic_stretched_x"
# =============================================================================
# engine_configuration_channel = EngineConfigurationChannel()
# =============================================================================
# =============================================================================
# env = UnityEnvironment(file_name=env_name, side_channels = [engine_configuration_channel])
# =============================================================================
env = UnityEnvironment(file_name=env_name, side_channels = [])

"""### Reset the environment
To reset the environment, simply call `env.reset()`. This method takes no argument and returns nothing but will send a signal to the simulation to reset.
"""

env.reset()
print("Env reset successfull")
# =============================================================================
# """### Behavior Specs
# 
# #### Get the Behavior Specs from the Environment
# """
# 
# # We will only consider the first Behavior
# behavior_name = list(env.behavior_specs)[0] 
# print(f"Name of the behavior : {behavior_name}")
# spec = env.behavior_specs[behavior_name]
# 
# """#### Get the Observation Space from the Behavior Specs"""
# 
# # Examine the number of observations per Agent
# print("Number of observations : ", len(spec.observation_shapes))
# 
# # Is there a visual observation ?
# # Visual observation have 3 dimensions: Height, Width and number of channels
# vis_obs = any(len(shape) == 3 for shape in spec.observation_shapes)
# print("Is there a visual observation ?", vis_obs)
# 
# """#### Get the Action Space from the Behavior Specs"""
# 
# # Is the Action continuous or multi-discrete ?
# if spec.is_action_continuous():
#   print("The action is continuous")
# if spec.is_action_discrete():
#   print("The action is discrete")
# 
# # How many actions are possible ?
# print(f"There are {spec.action_size} action(s)")
# 
# # For discrete actions only : How many different options does each action has ?
# if spec.is_action_discrete():
#   for action, branch_size in enumerate(spec.discrete_action_branches):
#     print(f"Action number {action} has {branch_size} different options")
# 
# """### Steping the environment
# 
# #### Get the steps from the Environment
# You can do this with the `env.get_steps(behavior_name)` method. If there are multiple behaviors in the Environment, you can call this method with each of the behavior's names.
# _Note_ This will not move the simulation forward.
# """
# 
# decision_steps, terminal_steps = env.get_steps(behavior_name)
# 
# """#### Set actions for each behavior
# You can set the actions for the Agents of a Behavior by calling `env.set_actions()` you will need to specify the behavior name and pass a tensor of dimension 2. The first dimension of the action must be equal to the number of Agents that requested a decision during the step.
# """
# 
# env.set_actions(behavior_name, spec.create_empty_action(len(decision_steps)))
# 
# """#### Move the simulation forward
# Call `env.step()` to move the simulation forward. The simulation will progress until an Agent requestes a decision or terminates.
# """
# 
# env.step()
# =============================================================================

"""### Observations

#### Show the observations for one of the Agents
`DecisionSteps.obs` is a tuple containing all of the observations for all of the Agents with the provided Behavior name.
Each value in the tuple is an observation tensor containing the observation data for all of the agents.
"""

# =============================================================================
# # Commented out IPython magic to ensure Python compatibility.
# import matplotlib.pyplot as plt
# # %matplotlib inline
# 
# for index, shape in enumerate(spec.observation_shapes):
#   if len(shape) == 3:
#     print("Here is the first visual observation")
#     plt.imshow(decision_steps.obs[index][0,:,:,:])
#     plt.show()
# =============================================================================

# =============================================================================
# for index, shape in enumerate(spec.observation_shapes):
#   if len(shape) == 1:
#     print("First vector observations : ", decision_steps.obs[index][0,:])
# =============================================================================

"""### Run the Environment for a few episodes"""
behavior_name = list(env.behavior_specs)[0] 
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]

count = 0       
for episode in range(100):
  env.reset()
  decision_steps, terminal_steps = env.get_steps(behavior_name)
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  
  while not done:
    # Track the first agent we see if not tracking 
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[0] 
  
# =============================================================================
#     for index, shape in enumerate(spec.observation_shapes):
#       if index == 2:
#         my_list = decision_steps.obs[index][0,:].tolist()
#         my_formatted_list = [ '%.2f' % elem for elem in my_list ]
#         print(my_formatted_list)
#         print("----------------------------------------------------------")
# =============================================================================
    
    visual_observation = decision_steps.obs[0][0,:]
    vector_observation = decision_steps.obs[2][0,:]
# =============================================================================
#     print(visual_observation)
# =============================================================================
    print(vector_observation)
    with h5py.File(H5FILE, 'r+') as f:
        vis_data = f['visual_obs']
        vis_data[count, :, :, :] = visual_observation
        vec_data = f['vector_obs']
        vec_data[count, :] = vector_observation
    # Move the simulation forward
    
    centre_ray = decision_steps.obs[1][0][2]
    right_ray = decision_steps.obs[1][0][5]
    left_ray = decision_steps.obs[1][0][8]

    turn_left = 1
    turn_right = 2
    turn_180 = 3
    hard_left = 4
    hard_right = 5

    print(left_ray, centre_ray, right_ray)
    # Generate an action for all agents
    action = spec.create_random_action(len(decision_steps))
 
    action[0][0]=1
    if left_ray==1.0 and right_ray==1.0 and centre_ray==1.0:
        action[0][1]= choice(a=[0,1,2], size=1, p=[0.4,0.3,0.3])
# =============================================================================
#         print("case 1 - no wall detected")
# =============================================================================
        
    elif left_ray< 1.0 and right_ray==centre_ray==1.0:
        action[0][1]= hard_right
        action[0][0]=0
# =============================================================================
#         print("case 2 - turn right")
# =============================================================================
        
    elif right_ray< 1.0 and left_ray==centre_ray==1.0:
        action[0][1]= hard_left
        action[0][0]=0
# =============================================================================
#         print("case 3 - turn left")
# =============================================================================
        
    elif right_ray==1.0 and left_ray==1.0 and centre_ray<1.0:
        action[0][1]= turn_180
        action[0][0]=0
# =============================================================================
#         print("case 4 - turn 180")
# =============================================================================
        
    elif right_ray<1.0 and left_ray<1.0 and centre_ray==1.0:
        action[0][1]= turn_180
        action[0][0]=0
# =============================================================================
#         print("case 5 - turn 180")
# =============================================================================
        
    elif right_ray<1.0 and centre_ray<1.0 and left_ray==1.0:
        action[0][1]= hard_left
        action[0][0]=0
# =============================================================================
#         print("case 6 - turn left")
# =============================================================================
        
    elif left_ray<1.0 and centre_ray<1.0 and right_ray==1.0:
        action[0][1]= hard_right
        action[0][0]=0
# =============================================================================
#         print("case 7 - turn right")
# =============================================================================
        
    # Set the actions
# =============================================================================
#     print("Action----", action)
# =============================================================================
    env.set_actions(behavior_name, action)

    env.step()

    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)

# =============================================================================
#     ray_left = decision_steps.obs[1][]
# =============================================================================
    count +=1
    print(count)

    if tracked_agent in decision_steps: # The agent requested a decision
      episode_rewards += decision_steps[tracked_agent].reward
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")

"""### Close the Environment to free the port it is using"""

env.close()
print("Closed environment")