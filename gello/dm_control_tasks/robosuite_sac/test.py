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

csv_folder = "/home/sj/Assistive_Feeding_Gello/csv/diffstartsamegoalsp100/pose5/test"
end_eff_folder = "/home/sj/Assistive_Feeding_Gello/csv/diffstartsamegoalsp100/end_eff5/test"

csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
end_eff_files = glob.glob(os.path.join(end_eff_folder, '*.csv'))

low_dim_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for csv_file, end_eff_file in zip(csv_files, end_eff_files):
        joint_data = pd.read_csv(csv_file, usecols=range(0, 6), engine='python').astype(np.float64)
        end_eff_pos_data = pd.read_csv(end_eff_file, usecols=range(0, 3), engine='python').astype(np.float64)
        end_eff_quat_data = pd.read_csv(end_eff_file, usecols=range(3, 7), engine='python').astype(np.float64)

        joint_data = torch.tensor(joint_data.values).float().to(device)
        end_eff_pos_data = torch.tensor(end_eff_pos_data.values).float().to(device)
        end_eff_quat_data = torch.tensor(end_eff_quat_data.values).float().to(device)

# obs = {"robot0_joint_pos": joint_data, "robot0_eef_pos": end_eff_pos_data, "robot0_eef_quat": end_eff_quat_data}
# print(obs["robot0_joint_pos"].shape, obs["robot0_eef_pos"].shape, obs["robot0_eef_quat"].shape)

obs = {"robot0_eef_pos": end_eff_pos_data}
print(obs["robot0_eef_pos"])

# ckpt_path = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/robomimic/bc_trained_models/allready2/20240607064247/models/model_epoch_100.pth"
ckpt_path = "/home/sj/Assistive_Feeding_Gello/robomimic/bc_trained_models/allready2/20240614134639/models/model_epoch_100.pth"
# Restore policy
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
