"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gym
from gym import spaces
import imageio


class RobocasaImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
            

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.obs_keys = low_dim_keys + image_keys
        # print(f"{self.image_keys=}")
        observation_space = spaces.Dict()
        # print(f"{shape_meta['obs']=}")
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 1
            elif key.endswith("state"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
            observation_space[key] = this_space
        # print(f"{observation_space=}")
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                # print(key, raw_obs[key].shape)
                # raw_obs[key] = raw_obs[key].transpose(2, 0, 1)
                if obs["rgb"] is None:
                    obs["rgb"] = raw_obs[key]
                else:
                    obs["rgb"] = np.concatenate(
                        [obs["rgb"], raw_obs[key]], axis=0
                    )  # C H W
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key]
                else:
                    obs["state"] = np.concatenate([obs["state"], raw_obs[key]], axis=-1)
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        obs["rgb"] *= 255  # [0, 1] -> [0, 255], in float64
        # print("&"*100)
        # print(obs["state"].shape)
        # print(obs["rgb"].shape)
        # print(raw_obs.keys())
        # print(self.obs_keys)
#         (9,)
# (6, 96, 96)
# (9,)
# (128, 3, 64)
# dict_keys(['agentview_image', 'robot0_eye_in_hand_image', 'object', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_base_to_eef_quat_site'])
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)
        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True
            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        for _ in range(3): # for action repeat
            raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
        
def create_shape_meta(img_size, include_state):
    shape_meta = {
    "obs": {
        "agentview_image": {
            # gym expects (H, W, C)
            "shape": [3, img_size, img_size],
            "type": "rgb",
        },
        "robot0_eye_in_hand_image": {
            # gym expects (H, W, C)
            "shape": [3, img_size, img_size],
            "type": "rgb",
        },
    },
    "action": {"shape": [12]},
    }
    # if include_state:
    #     shape_meta["obs"].update(STATE_SHAPE_META)
    return shape_meta
    
def sanitize_for_robomimic(config):
    if "layout_ids" in config:
        del config["layout_ids"]
    if "style_ids" in config:
        del config["style_ids"]
    if "obj_groups" in config:
        del config["obj_groups"]
    if "translucent_robot" in config:
        del config["translucent_robot"]
    if "obj_instance_split" in config:
        del config["obj_instance_split"]
    return config

def get_env_details(config, suite, task):
    import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
    
    dataset_path = "/share/portal/sk3428/dppo_irl/Data/robocasa_datasets/bread/image_64_shaped_done1_v141.hdf5"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    if task.lower() in ["stack", "door", "bread"]:
        env_meta["env_kwargs"] = sanitize_for_robomimic(env_meta["env_kwargs"])

    shape_meta = create_shape_meta(
        img_size=config.img_size,
        include_state=True,
    )
    return dataset_path, env_meta, shape_meta


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robocasa/finetune/bread/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    # import robomimic.utils.env_utils as EnvUtils
    # import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt

    wrappers = cfg.env.wrappers
    
    import robocasa
    import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
    import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

    wrappers = cfg.env.wrappers
    _, env_meta, shape_meta = get_env_details(cfg, "robocasa", "bread")
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False
    
    # obs_modality_dict = {
    #     "low_dim": (
    #         wrappers.robomimic_image.low_dim_keys
    #         if "robomimic_image" in wrappers
    #         else wrappers.robomimic_lowdim.low_dim_keys
    #     ),
    #     "rgb": (
    #         wrappers.robomimic_image.image_keys
    #         if "robomimic_image" in wrappers
    #         else None
    #     ),
    # }
    # if obs_modality_dict["rgb"] is None:
    #     obs_modality_dict.pop("rgb")
    # ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    # with open(cfg.robomimic_env_cfg_path, "r") as f:
    #     env_meta = json.load(f)
    # env = EnvUtils.create_env_from_metadata(
    #     env_meta=env_meta,
    #     render=False,
    #     render_offscreen=False,
    #     use_image_obs=True,
    # )
    # env.env.hard_reset = False

    wrapper = RobocasaImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["robot0_eye_in_hand_image"],
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys())
    img = wrapper.render()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test.png")
