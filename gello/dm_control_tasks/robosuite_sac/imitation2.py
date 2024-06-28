import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy
import glob 

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

import pandas as pd
import numpy as np

csv_folder = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/csv/samestartdiffgoalsp100/pose5/test"
end_eff_folder = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/csv/samestartdiffgoalsp100/end_eff5/test"

csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
end_eff_files = glob.glob(os.path.join(end_eff_folder, '*.csv'))

low_dim_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for csv_file, end_eff_file in zip(csv_files, end_eff_files):
        joint_data = pd.read_csv(csv_file, usecols=range(0, 6), engine='python').astype(np.float64)
        end_eff_pos_data = pd.read_csv(end_eff_file, usecols=range(0, 3), engine='python').astype(np.float64)
        end_eff_quat_data = pd.read_csv(end_eff_file, usecols=range(3, 7), engine='python').astype(np.float64)

        joint_data = torch.tensor(joint_data.values).float()
        joint_data = joint_data.to(device)
        # joint_data = joint_data.unsqueeze(0)
        # joint_data = joint_data.view(30, 1, 6)
        # joint_data = joint_data.squeeze(0)
        

obs = {"robot0_joint_pos": joint_data}

print(obs["robot0_joint_pos"].shape)
print(joint_data)

# ckpt_path = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/robomimic/bc_trained_models/allready3/20240607140542/models/model_epoch_100.pth"
ckpt_path = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/robomimic/bc_trained_models/allready/20240607063910/models/model_epoch_150.pth"
# Restore policy
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

# print(obs["robot0_joint_pos"])

# action = policy(obs)
# print(action)

for i in range(len(obs["robot0_joint_pos"])):
    obs_to_pass = {"robot0_joint_pos": obs["robot0_joint_pos"][i]}
    print(obs_to_pass["robot0_joint_pos"])
    action = policy(obs_to_pass)
    print(action)