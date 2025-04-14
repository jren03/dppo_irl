import os
import numpy as np


normalization_save_path = os.path.join(
    "data/robomimic/custom_normalization", "can_normalization.npz"
)
np.savez_compressed(
    normalization_save_path,
    obs_min=np.array([-0.07376085, -0.42253869,  0.86021113,  0.59080709, -0.73516192, -0.22861566, -0.34267776,  0.0154101,  -0.0403102 ]),
    obs_max=np.array([ 0.29611527,  0.41260722,  1.22413338,  0.99997648,  0.47231822,  0.13808564, 0.08067435,  0.04106108, -0.01176225]),
    action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
)


# Yunhai
# np.savez_compressed(
#     normalization_save_path,
#     obs_min=np.array([-0.3, -0.5, 0.8, -1.0, -1.0, -1.0, -1.0, -0.05, -0.05]),
#     obs_max=np.array([ 0.3,  0.5, 1.2,  1.0,  1.0,  1.0,  1.0,  0.05,  0.05]),
#     action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
#     action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0]),
# )



