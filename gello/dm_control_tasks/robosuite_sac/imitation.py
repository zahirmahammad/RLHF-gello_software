import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import os
import torch
import numpy as np
import robomimic.utils.file_utils as FileUtils

class CSVDataset(Dataset):
    def __init__(self, csv_file, end_eff_file):
        self.joint_data = pd.read_csv(csv_file, usecols=range(0, 6), engine='python').astype(np.float64)
        self.end_eff_pos_data = pd.read_csv(end_eff_file, usecols=range(0, 3), engine='python').astype(np.float64)
        self.end_eff_quat_data = pd.read_csv(end_eff_file, usecols=range(3, 7), engine='python').astype(np.float64)
        # self.data_frame = pd.concat([self.joint_data, self.end_eff_pos_data, self.end_eff_quat_data], axis=1)
        # self.jointdf = torch.tensor(self.joint_data.values, dtype=torch.float32)
        # self.end_eff_pos_df = torch.tensor(self.end_eff_pos_data.values, dtype=torch.float32)
        # self.end_eff_quat_df = torch.tensor(self.end_eff_quat_data.values, dtype=torch.float32)
        self.jointdf = np.array(self.joint_data.values, dtype=np.float32)
        self.end_eff_pos_df = np.array(self.end_eff_pos_data.values, dtype=np.float32)
        self.end_eff_quat_df = np.array(self.end_eff_quat_data.values, dtype=np.float32)
        self.data_frame = np.concatenate([self.jointdf, self.end_eff_pos_df, self.end_eff_quat_df], axis=1)
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        joint_pos = self.jointdf[idx]
        eef_pos = self.end_eff_pos_df[idx]
        eef_quat = self.end_eff_quat_df[idx]
        # obs_dict = {
        #     "robot0_joint_pos": joint_pos,
        #     "robot0_eef_pos": eef_pos,
        #     "robot0_eef_quat": eef_quat
        # }
        obs_dict = {
            "robot0_joint_pos":self.jointdf[idx]
        }
        return obs_dict

def load_datasets(csv_folder, end_eff_folder):
    all_datasets = []
    for csv_file, end_eff_file in zip(glob.glob(os.path.join(csv_folder, '*.csv')), glob.glob(os.path.join(end_eff_folder, '*.csv'))):
        dataset = CSVDataset(csv_file=csv_file, end_eff_file=end_eff_file)
        all_datasets.append(dataset)
    return ConcatDataset(all_datasets)

csv_folder = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/csv/samestartdiffgoalsp100/pose5/test"
end_eff_folder = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/csv/samestartdiffgoalsp100/end_eff5/test"

loaded = load_datasets(csv_folder=csv_folder, end_eff_folder=end_eff_folder)
loader = DataLoader(loaded, batch_size=10, shuffle=True)  # Adjust batch_size as needed

print(loaded.cumulative_sizes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ckpt_path = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/robomimic/bc_trained_models/allready3/20240607140542/models/model_epoch_100.pth"
ckpt_path = "/home/sj/assistive_feed_gello/Assistive_Feeding_Gello/robomimic/bc_trained_models/allready/20240607063910/models/model_epoch_150.pth"
# Restore policy
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

for batch in loader:
    # joints = batch["robot0_joint_pos"].to(device)
    # end_eff_pos = batch["robot0_eef_pos"].to(device)
    # end_eff_quat = batch["robot0_eef_quat"].to(device)
    joints = batch["robot0_joint_pos"].to(device)
    with torch.no_grad():
        # joints = joints.view(10, 1, 6)
        # Ensure inputs are compatible with the policy
        # obs = np.array({"robot0_joint_pos": joints, "robot0_eef_pos": end_eff_pos, "robot0_eef_quat": end_eff_quat})
        # obs = {"robot0_joint_pos": joints, "robot0_eef_pos": end_eff_pos, "robot0_eef_quat": end_eff_quat}
        obs = {"robot0_joint_pos": joints}
        outputs = policy(obs)
        print(outputs)
