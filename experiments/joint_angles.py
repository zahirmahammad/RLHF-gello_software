import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot

import pandas as pd


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    # robot_port: int = 50003  # for trajectory
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"

    #---- Hardware ---
    # hostname: str = "192.168.77.243"
    # robot_ip: str = "192.168.77.21"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False

def print_state(env):

    # while True:
    obs = env.get_obs()["joint_positions"]
    return obs
        # time.sleep(1)
        # print("Observation: ", obs)
        # moved = env.step(np.array([0, -1.57, 0, -1.57, 0, 0]))


def execute_trajectory(env, csv_file_path):
    data = pd.read_csv(csv_file_path)
    # Convert angles to lists of lists
    joint_angles = data[['shoulder_pan_angle', 'shoulder_lift_angle','elbow_angle', 'wrist1_angle', 'wrist2_angle', 'wrist3_angle']].values.tolist()
    print(joint_angles)
    # joint_angles.append([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    for angles in joint_angles:
        # Set the joint angles
        time.sleep(0.8)
        moved = env.step(np.array(angles))
        print(moved["joint_positions"])
        

if __name__ == "__main__":
    robot_client = ZMQClientRobot(port=Args.robot_port, host=Args.hostname)
    camera_clients = {
        # you can optionally add camera nodes here for imitation learning purposes
        # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
        # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }
    csv_file_path =  Path(__file__).parent.parent / "csv" / "tofu2_10_new.csv"

    env = RobotEnv(robot_client, control_rate_hz=Args.hz, camera_dict=camera_clients)

    execute_trajectory(env, csv_file_path)
    # print(print_state(env))
    # main(tyro.cli(Args))




    

    

