import os
import json

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)
    
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
                
def make_async(
    config,
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    **kwargs,
):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment.

    asynchronous : bool
        If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses
        `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.

    wrappers : dictionary, optional
        Each key is a wrapper class, and each value is a dictionary of arguments

    Returns
    -------
    :class:`gym.vector.VectorEnv`
        The vectorized environment.

    Example
    -------
    >>> env = gym.vector.make('CartPole-v1', num_envs=3)
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """

    if env_type == "furniture":
        from furniture_bench.envs.observation import DEFAULT_STATE_OBS
        from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
        from env.gym_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper

        env = FurnitureRLSimEnv(
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture=furniture,
            gpu_id=gpu_id,
            headless=headless,
            num_envs=num_envs,
            observation_space="state",
            randomness=randomness,
            max_env_steps=max_episode_steps,
            record=record,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env = FurnitureRLSimEnvMultiStepWrapper(
            env,
            n_obs_steps=obs_steps,
            n_action_steps=act_steps,
            prev_action=False,
            reset_within_step=False,
            pass_full_observations=False,
            normalization_path=normalization_path,
            sparse_reward=sparse_reward,
        )
        return env

    # avoid import error due incompatible gym versions
    from gym import spaces
    from env.gym_utils.async_vector_env import AsyncVectorEnv
    from env.gym_utils.sync_vector_env import SyncVectorEnv
    from env.gym_utils.wrapper import wrapper_dict

    __all__ = [
        "AsyncVectorEnv",
        "SyncVectorEnv",
        "VectorEnv",
        "VectorEnvWrapper",
        "make",
    ]

    # import the envs
    if robomimic_env_cfg_path is not None:
        if env_type == "robocasa":
            import robocasa
            import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
            import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
            import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
        else:
            import robomimic.utils.env_utils as EnvUtils
            import robomimic.utils.obs_utils as ObsUtils
    elif "avoiding" in id:
        import gym_avoiding
    else:
        import d4rl.gym_mujoco
    from gym.envs import make as make_

    def _make_env():
        if robomimic_env_cfg_path is not None:
            if env_type == "robocasa":
                # if render_offscreen or use_image_obs:
                #     os.environ["MUJOCO_GL"] = "egl"
                
                # with open(robomimic_env_cfg_path, "r") as f:
                #     env_meta = json.load(f)
                # _, env_meta, shape_meta = get_env_details(config, "robocasa", id)
                # env_meta["reward_shaping"] = reward_shaping
                # env = EnvUtils.create_env_for_data_processing(
                #     env_meta=env_meta,
                #     camera_names=["agentview", "robot0_eye_in_hand"],
                #     camera_height=config.img_size,
                #     camera_width=config.img_size,
                #     reward_shaping=True,
                # )
                # env.env.hard_reset = False
                obs_modality_dict = {
                    "low_dim": (
                        wrappers.robocasa_image.low_dim_keys
                        # if "robocasa_image" in wrappers
                        # else wrappers.robomimic_lowdim.low_dim_keys
                    ),
                    "rgb": (
                        wrappers.robocasa_image.image_keys
                        if "robocasa_image" in wrappers
                        else None
                    ),
                }
                if obs_modality_dict["rgb"] is None:
                    obs_modality_dict.pop("rgb")
                ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
                if render_offscreen or use_image_obs:
                    os.environ["MUJOCO_GL"] = "egl"
                with open(robomimic_env_cfg_path, "r") as f:
                    env_meta = json.load(f)
                env_meta["reward_shaping"] = reward_shaping
                env_meta["env_kwargs"]["camera_heights"] = config.img_size
                env_meta["env_kwargs"]["camera_widths"] = config.img_size
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    render=render,
                    # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
                    render_offscreen=render_offscreen,
                    use_image_obs=use_image_obs,
                    # render_gpu_device_id=0,
                )
                
                # env = EnvUtils.create_env_for_data_processing(
                #     env_meta=env_meta,
                #     camera_names=["agentview", "robot0_eye_in_hand"],
                #     camera_height=config.img_size,
                #     camera_width=config.img_size,
                #     reward_shaping=True,
                # )
                # Robosuite's hard reset causes excessive memory consumption.
                # Disabled to run more envs.
                # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
                env.env.hard_reset = False
            else:
                obs_modality_dict = {
                    "low_dim": (
                        wrappers.robomimic_image.low_dim_keys
                        if "robomimic_image" in wrappers
                        else wrappers.robomimic_lowdim.low_dim_keys
                    ),
                    "rgb": (
                        wrappers.robomimic_image.image_keys
                        if "robomimic_image" in wrappers
                        else None
                    ),
                }
                if obs_modality_dict["rgb"] is None:
                    obs_modality_dict.pop("rgb")
                ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
                if render_offscreen or use_image_obs:
                    os.environ["MUJOCO_GL"] = "egl"
                with open(robomimic_env_cfg_path, "r") as f:
                    env_meta = json.load(f)
                env_meta["reward_shaping"] = reward_shaping
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    render=render,
                    # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
                    render_offscreen=render_offscreen,
                    use_image_obs=use_image_obs,
                    # render_gpu_device_id=0,
                )
                # Robosuite's hard reset causes excessive memory consumption.
                # Disabled to run more envs.
                # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
                env.env.hard_reset = False
        else:  # d3il, gym
            if "kitchen" not in id:  # d4rl kitchen does not support rendering!
                kwargs["render"] = render
            env = make_(id, **kwargs)

        # add wrappers
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env

    def dummy_env_fn():
        """TODO(allenzren): does this dummy env allow camera obs for other envs besides robomimic?"""
        import gym
        import numpy as np
        from env.gym_utils.wrapper.multi_step import MultiStep

        # Avoid importing or using env in the main process
        # to prevent OpenGL context issue with fork.
        # Create a fake env whose sole purpose is to provide
        # obs/action spaces and metadata.
        env = gym.Env()
        observation_space = spaces.Dict()
        if shape_meta is not None:  # rn only for images
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = -1, 1
                elif key.endswith("state"):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        else:
            observation_space["state"] = gym.spaces.Box(
                -1,
                1,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        env.observation_space = observation_space
        env.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.int64)
        env.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": 12,
        }
        return MultiStep(env=env, n_obs_steps=wrappers.multi_step.n_obs_steps)

    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(
            env_fns,
            dummy_env_fn=(
                dummy_env_fn if render or render_offscreen or use_image_obs else None
            ),
            delay_init="avoiding" in id,  # add delay for D3IL initialization
        )
        if asynchronous
        else SyncVectorEnv(env_fns)
    )
