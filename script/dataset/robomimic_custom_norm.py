import os
import numpy as np


normalization_save_path = os.path.join(
    "data/robocasa/custom_normalization", "normalization.npz"
)
np.savez_compressed(
    normalization_save_path,
    obs_min=np.array([-0.3, -0.5, 0.8, -1.0, -1.0, -1.0, -1.0, -0.05, -0.05]),
    obs_max=np.array([ 0.3,  0.5, 1.2,  1.0,  1.0,  1.0,  1.0,  0.05,  0.05]),
    action_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    action_max=np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0]),
)