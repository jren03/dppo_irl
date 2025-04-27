import os
import numpy as np


########################################## For printing out obs_min/obs_max ##########################################
import h5py
import numpy as np
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

# Path to dataset
file_path = "/share/portal/sk3428/dppo_irl/Data/robocasa_datasets/nutassemblyround/image_96_shaped_done1_v141.hdf5"
# "/share/portal/vm392/diffusion_dreamer/robocasa_datasets/bread/image_96_shaped_done1_v141.hdf5"
env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=file_path)
print(env_meta)

# Observation keys of interest
low_dim_obs_names = [
    "robot0_eef_pos",      # 3D position
    "robot0_eef_quat",     # 4D quaternion
    "robot0_gripper_qpos"  # 2D gripper state
]

# Initialize lists to store all observations
all_obs = []

# Open HDF5 file
with h5py.File(file_path, "r") as f:
    demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]  # Find all demos
    print(f"Found {len(demo_keys)} demos.")

    for demo_name in demo_keys:
        demo = f[f"data/{demo_name}/obs"]  # Access observations

        # Collect observations for this demo
        obs_concat = np.hstack([demo[obs_name][:] for obs_name in low_dim_obs_names])  # Stack per timestep
        all_obs.append(obs_concat)  # Store for min/max computation

# Stack all collected data
all_obs = np.vstack(all_obs)  # Shape (Total_timesteps, 9)

# Compute min and max
obs_min = np.min(all_obs, axis=0)
obs_max = np.max(all_obs, axis=0)

print("obs_min =", obs_min)
print("obs_max =", obs_max)


normalization_save_path = os.path.join(
    "data/robocasa/custom_normalization", "nutassemblyround_normalization.npz" # Change the file path
)
np.savez_compressed(
    normalization_save_path,
    obs_min=obs_min,
    obs_max=obs_max,
    action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0]),
)


