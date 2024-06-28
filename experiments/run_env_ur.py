import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import csv

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    # robot_port: int = 6001 #for_mujoco
    robot_port: int = 50003  # for trajectory
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    # hostname: str = "127.0.0.1" #for_mujoco
    hostname: str = "192.168.77.243"
    robot_ip: str = "192.168.77.21" 
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                print('in if condition')
                reset_joints = (
                    # [1.57, -1.57, -1.57, -1.57, 1.57, 0.0]
                    [-1.57, -1.57, -1.57, -1.57, 1.57, 1.57, 0.0]
                )  # Change this to your own reset joints
            else:
                reset_joints = np.array(args.start_joints)
            agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
            curr_joints = env.get_obs()["joint_positions"]
            
            print("Current Joints", curr_joints)
            # print("Reset Joints", reset_joints)
            # print("reset_joints type", type(reset_joints))

            # curr_joints = np.array(curr_joints)
            print("curr_joints type", type(curr_joints))
            if len(reset_joints) == len(curr_joints):
                max_delta = (np.abs(curr_joints - np.array(reset_joints))).max()
                # print("reset joints_now", reset_joints)
                print("max_delta", max_delta)   
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    # here we only care about the first 6 joints
    # start_pos = start_pos[:6]
    obs = env.get_obs()  # gets the gello joint positions
    joints = obs["joint_positions"] # gets the UR5e joint positions

    print("start_pos", (start_pos))

    # print("start_pos type", type(start_pos))
    # joints = np.array(joints)
    # print("joints", np.rad2deg(joints))
    print("joints", joints)
    # print("joints type", type(joints))

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)
    print("id_max_joint_delta", id_max_joint_delta)
    max_joint_delta = 0.8
    print(abs_deltas[id_max_joint_delta])
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        # print("id_mask", id_mask)
        ids = np.arange(len(id_mask))[id_mask]
        # print("ids", ids)
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        print("getting obs to go to start position")
        obs = env.get_obs()
        print("got obs", obs)
        command_joints = agent.act(obs)
        # here we only care about the first 6 joints
        # command_joints = command_joints[:6]
        current_joints = obs["joint_positions"]
        print("command_joints", command_joints)
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            print("max_joint_delta", max_joint_delta)
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    # here we only care about the first 6 joints
    # joints = joints[:6]
    action = agent.act(obs) # gets the gello joint positions
    # here we only care about the first 6 joints
    # action = action[:6]
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    start_time_print = time.time()
    while True:
            gello_angle = agent.act(obs)
        # if (-0.523599 > gello_angle[0] > -2.61799) and (-0.785398 > gello_angle[1] > -2.0944) and  (-0.523599 > gello_angle[2] > -2.61799) and (-0.523599 > gello_angle[3] > -2.61799):
            num = time.time() - start_time_print
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            action = agent.act(obs)
            # here we only care about the first 6 joints
            # action = action[:6]
            dt = datetime.datetime.now()
            if args.use_save_interface:
                state = kb_interface.update()
                if state == "start":
                    dt_time = datetime.datetime.now()
                    save_path = (
                        Path(args.data_dir).expanduser()
                        / args.agent
                        / dt_time.strftime("%m%d_%H%M%S")
                    )
                    save_path.mkdir(parents=True, exist_ok=True)
                    print(f"Saving to {save_path}")
                elif state == "save":
                    assert save_path is not None, "something went wrong"
                    save_frame(save_path, dt, obs, action)
                elif state == "normal":
                    save_path = None
                else:
                    raise ValueError(f"Invalid state {state}")
            obs = env.step(action)
            csv_file_path = '/home/sj/RLHF-gello_software/csv/output21.csv'      # Writing to CSV file
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['shoulder_pan_angle', 'shoulder_lift_angle', 'elbow_angle', 'wrist1_angle', 'wrist2_angle', 'wrist3_angle', 'end_eff_x', 'end_eff_y', 'end_eff_z', 'end_eff_xq', 'end_eff_yq', 'end_eff_zq', 'end_eff_w', 'gripper_pos'])  # Write the header    writer.writerows(data)
                obs = env.get_obs()["joint_positions"]

                obs_end_eff = env.get_obs()["ee_pos_quat"]

                # gripper_pos = env.get_obs()["gripper_position"]

                obs_combined = np.concatenate((obs, obs_end_eff))

                writer.writerow(obs_combined)
                
if __name__ == "__main__":
    main(tyro.cli(Args))
