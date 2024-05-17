import os
import time
import sys
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from network import *
from buffer import *
from td3_torch import *

if __name__=="__main__":
    print("TD3 Envi")

    td3_dir = os.path.join("td3")

    if not os.path.exists(td3_dir):
        os.makedirs(td3_dir)

    print("TD3 Dir: ", td3_dir)

    env_name = "Wipe"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        horizon=200,
        render_camera="frontview",
        has_offscreen_renderer=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    ####

    # critic_network = CriticNetwork(input_dims=[8], n_actions=8)

    # actor_network = ActorNetwork(input_dims=[8], n_actions=8)

    # replay_buffer = ReplayBuffer(max_size=8, input_shape=[8], n_actions=8)

    ####

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, input_dims=env.observation_space.shape, 
            tau=0.005, env=env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size, n_actions=env.action_space.shape[0])

    writer = SummaryWriter("logs")
    n_games = 10000
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate: {actor_learning_rate} critic_learning_rate: {critic_learning_rate} batch_size: {batch_size} layer1_size: {layer1_size} layer2_size: {layer2_size}"

    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            # print(env.step(action))
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if (i % 10):
            agent.save_models()

        print(f"Episode {i}, Score: {score}")
