import os
import time
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

if __name__=="__main__":
    print("BC Envi")

    if not os.path.exists("robosuite_sac/td3"):
        os.makedirs("robosuite_sac/td3")
        print("Made directory")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["UR5e"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)
