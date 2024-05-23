import h5py
import json
import numpy as np
import pandas as pd
import glob 
import os

# Create sample data
csv_folder = "/home/sj/Downloads/csv"
low_dim_data = []
next_low_dim_data = []
for csv_file in glob.glob(os.path.join(csv_folder, '*.csv')):
    dataset = pd.read_csv(csv_file, usecols=range(10, 16), skiprows=1, engine='python')
    next_dataset = pd.read_csv(csv_file, usecols=range(10, 16), skipfooter=1, engine='python')
    low_dim_data.append(dataset)
    next_low_dim_data.append(next_dataset)

print("Low dim data shape:", low_dim_data[0].shape)
# print("Low dim data shape:", np.array(low_dim_data[0][1:]))


# Define the data
total_samples = len(low_dim_data)
env_args = {
    "env_name": "Lift",
    "type": 1,
    "env_kwargs": {
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
        "robots": ["Panda"],
        "camera_depths": True,
        "camera_heights": 84,
        "camera_widths": 84,
        "render_gpu_device_id": 0,
        "reward_shaping": False,
        "camera_names": ["agentview", "robot0_eye_in_hand"]
    },
    "env_version": "1.4.1",
}

# Define HDF5 file path
file_path = "/home/sj/Downloads/data.hdf5"

# if not os.path.exists(file_path):
#     os.makedirs(file_path)
# Delete the existing HDF5 file if it exists
if os.path.exists(file_path):
    os.remove(file_path)

# fk = ForwardKinematicsUR5e()

# Write data to HDF5 file
with h5py.File(file_path, "w") as f:
    # Create the data group
    data_group = f.create_group("data")
    
    # Write attributes for the data group
    data_group.attrs["total"] = total_samples
    data_group.attrs["env_args"] = json.dumps(env_args)

    for i in range(len(low_dim_data)):
        rewards_data = [0]*(len(low_dim_data[i])-3) + [1]*3
        globals()[f'states_demo_{i}'] = np.array(low_dim_data[i])
        globals()[f'actions_demo_{i}'] = np.random.uniform(-1, 1, (len(low_dim_data[i]), 6))
        globals()[f'rewards_demo_{i}'] = rewards_data
        globals()[f'dones_demo_{i}'] = rewards_data
        globals()[f'obs_demo_{i}'] = np.array(low_dim_data[i])
        globals()[f'next_obs_demo_{i}'] = np.array(low_dim_data[i])
        # object_demo_
        num_rows = low_dim_data[i].shape[0]
        num_cols = 10

        # Define the ranges for each column
        column_ranges = [
            (0.02137, 0.024942),
            (0.024942, 0.026981),
            (0.86433, 0.83142),
            (-0.0004, 0),
            (-0.002, 0),
            (0.9719, 0.96911),
            (0.23536, 0.24663),
            (0.0061206, -0.12043),
            (-0.0047403, -0.04022),
            (0.0039, 0.19099)
        ]
        
        # Initialize the array
        object = np.zeros((num_rows, num_cols))

        # Fill each column with random values within the specified ranges
        for col in range(num_cols):
            low, high = column_ranges[col]
            if low > high:  # Ensure the correct range order
                low, high = high, low
            object[:, col] = np.random.uniform(low=low, high=high, size=num_rows)

        print("Actions shape:", np.array(globals()[f'actions_demo_{i}']).shape)

        # Write data for the current trajectory
        demo_group = data_group.create_group(f"demo_{i}")
        demo_group.attrs["num_samples"] = low_dim_data[i].shape[0]
        demo_group.attrs["model_file"] = "XML string"  # Example value for demonstration
        demo_group.create_dataset("states", data=locals()[f"states_demo_{i}"])
        demo_group.create_dataset("actions", data=locals()[f"actions_demo_{i}"])
        demo_group.create_dataset("rewards", data=locals()[f"rewards_demo_{i}"])
        demo_group.create_dataset("dones", data=locals()[f"dones_demo_{i}"])


        # Write observation data directly
        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("robot0_joint_pos", data=np.array(low_dim_data[i]))
        obs_group.create_dataset("robot0_eef_pos", data=np.random.uniform(-1, 1, (num_rows, 3)))  
        obs_group.create_dataset("robot0_eef_quat", data=np.random.uniform(-1, 1, (num_rows, 4)))  
        obs_group.create_dataset("robot0_gripper_qpos", data=np.random.uniform(-1, 1, (num_rows, 1)))  
        obs_group.create_dataset("object", data=object)
        
        next_obs_group = demo_group.create_group("next_obs")
        next_obs_group.create_dataset("robot0_joint_pos", data=np.array(next_low_dim_data[i]))
        next_obs_group.create_dataset("robot0_eef_pos", data=np.random.uniform(-1, 1, (num_rows, 3))) 
        next_obs_group.create_dataset("robot0_eef_quat", data=np.random.uniform(-1, 1, (num_rows, 4)))  
        next_obs_group.create_dataset("robot0_gripper_qpos", data=np.random.uniform(-1, 1, (num_rows, 1))) 
        next_obs_group.create_dataset("object", data=object)
    

print("Data has been written to:", file_path)

