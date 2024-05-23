import h5py
import json
import numpy as np
import pandas as pd
import glob 
import os
# from joblib import Parallel, delayed
# from forward_kinematics import ForwardKinematicsUR5e

# Create sample data
# csv_folder = "/home/sj/Downloads/csv"
csv_folder = "/home/sj/gello_software/csv"
low_dim_data = []
action_dataset = []
robot_end_eff_pos = []
robot_end_eff_quat = []
states_data = []
for csv_file in glob.glob(os.path.join(csv_folder, '*.csv')):
    dataset = pd.read_csv(csv_file, usecols=range(0, 5), header=0)
    action_dataframe = pd.read_csv(csv_file, usecols=range(6, 11), header=0)
    robot_end_eff_pos_df = pd.read_csv(csv_file, usecols=range(6, 8), header=0)
    robot_end_eff_quat_df = pd.read_csv(csv_file, usecols=range(9, 12), header=0)
    states_dataframe = pd.read_csv(csv_file, usecols=range(0, 12), header=0)

print(states_dataframe.head())
print(states_dataframe.shape)