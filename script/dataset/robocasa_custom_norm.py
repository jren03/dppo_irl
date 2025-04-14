import os
import numpy as np


normalization_save_path = os.path.join(
    "data/robocasa/custom_normalization", "cereal_normalization.npz"
)
np.savez_compressed(
    normalization_save_path,
    obs_min=np.array([-0.16663205, -0.42519651,  0.93681863,  0.53520979, -0.61286884, -0.18686484, -0.19187748,  0.01236833, -0.04035082]),
    obs_max=np.array([0.20531514,  0.46930787,  1.29175893,  0.99995052,  0.83770579,  0.28691145, 0.12117586,  0.04023938, -0.00668416]),
    action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0]),
)

# import h5py

# # Path to the dataset file
# file_path = "/share/portal/sk3428/dppo_irl/Data/robocasa_datasets/bread/image_64_shaped_done1_v141.hdf5"

# # Open the HDF5 file
# with h5py.File(file_path, "r") as f:
#     # Inspect top-level keys
#     print("Top-level keys:", list(f.keys()))

#     # Iterate through available demos
#     demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
#     print(f"Found {len(demo_keys)} demos:", demo_keys[:5])  # Print first few demo names

#     # Access a specific demo
#     demo_name = demo_keys[0]  # Pick the first demo
#     demo = f["data"][demo_name]

#     # Print demo-level attributes
#     obs = demo["obs"]
#     print("Observation Keys:", list(obs.keys()))
#     low_dim_obs_names = [
#             "robot0_eef_pos",
#             "robot0_eef_quat",
#             "robot0_gripper_qpos",
#         ]
#     for low_dim_obs_name in low_dim_obs_names:
#         dim = f[f"data/demo_0/obs/{low_dim_obs_name}"].shape[1]
#         print(low_dim_obs_name, dim)
#         print(f[f"data/demo_0/obs/{low_dim_obs_name}"])

########################################### For printing out obs_min/obs_max ##########################################
# import h5py
# import numpy as np
# import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

# # Path to dataset
# file_path = "/share/portal/sk3428/dppo_irl/Data/robocasa_datasets/cereal/image_96_shaped_done1_v141.hdf5"
# # "/share/portal/vm392/diffusion_dreamer/robocasa_datasets/bread/image_96_shaped_done1_v141.hdf5"
# env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=file_path)
# print(env_meta)

# # Observation keys of interest
# low_dim_obs_names = [
#     "robot0_eef_pos",      # 3D position
#     "robot0_eef_quat",     # 4D quaternion
#     "robot0_gripper_qpos"  # 2D gripper state
# ]

# # Initialize lists to store all observations
# all_obs = []

# # Open HDF5 file
# with h5py.File(file_path, "r") as f:
#     demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]  # Find all demos
#     print(f"Found {len(demo_keys)} demos.")

#     for demo_name in demo_keys:
#         demo = f[f"data/{demo_name}/obs"]  # Access observations

#         # Collect observations for this demo
#         obs_concat = np.hstack([demo[obs_name][:] for obs_name in low_dim_obs_names])  # Stack per timestep
#         all_obs.append(obs_concat)  # Store for min/max computation

# # Stack all collected data
# all_obs = np.vstack(all_obs)  # Shape (Total_timesteps, 9)

# # Compute min and max
# obs_min = np.min(all_obs, axis=0)
# obs_max = np.max(all_obs, axis=0)

# print("obs_min =", obs_min)
# print("obs_max =", obs_max)


