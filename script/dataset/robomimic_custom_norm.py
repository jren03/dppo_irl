# import os
# import numpy as np


# normalization_save_path = os.path.join(
#     "data/robomimic/custom_normalization", "can_normalization.npz"
# )
# np.savez_compressed(
#     normalization_save_path,
#     obs_min=np.array([-0.07376085, -0.42253869,  0.86021113,  0.59080709, -0.73516192, -0.22861566, -0.34267776,  0.0154101,  -0.0403102 ]),
#     obs_max=np.array([ 0.29611527,  0.41260722,  1.22413338,  0.99997648,  0.47231822,  0.13808564, 0.08067435,  0.04106108, -0.01176225]),
#     action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
#     action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
# )


# # Yunhai
# # np.savez_compressed(
# #     normalization_save_path,
# #     obs_min=np.array([-0.3, -0.5, 0.8, -1.0, -1.0, -1.0, -1.0, -0.05, -0.05]),
# #     obs_max=np.array([ 0.3,  0.5, 1.2,  1.0,  1.0,  1.0,  1.0,  0.05,  0.05]),
# #     action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
# #     action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0]),
# # )



import os
import numpy as np


########################################## For printing out obs_min/obs_max ##########################################
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# Path to dataset
file_path = "/share/portal/sk3428/dppo_irl/Data/robomimic_datasets/square/image_96_shaped_done1_v141.hdf5"
# "/share/portal/vm392/diffusion_dreamer/robocasa_datasets/bread/image_96_shaped_done1_v141.hdf5"
# env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=file_path)
# print(env_meta)

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
    "data/robomimic/custom_normalization", "square_normalization.npz" # Change the file path
)
np.savez_compressed(
    normalization_save_path,
    obs_min=obs_min,
    obs_max=obs_max,
    action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
)
