# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
'''
hydra_args 是 argparse 模块解析命令行参数后剩余的未知参数列表。
在代码中，parser.parse_known_args() 方法用于解析命令行参数，它会返回两个值：
args_cli：包含已识别的命令行参数的命名空间对象。
hydra_args：包含未被 argparse 解析器识别的命令行参数的列表。
这些未知参数通常是为了后续使用 Hydra 配置框架而保留的。Hydra 是一个用于动态配置应用程序的 Python 库，
它允许用户通过命令行或配置文件来灵活地配置应用程序的各种参数。
'''
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
# sys.argv 被重置为只包含脚本名称和 hydra_args，这样可以确保 Hydra 在后续处理时只接收这些未知参数，
# 从而避免与 argparse 已经处理过的参数冲突。
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml, load_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# 配置入口点可以是YAML文件或Python配置类,
# "sb3_cfg_entry_point"这个是指定了register中的entry_point，
# 具体看：注册环境章节（https://docs.robotsfan.com/isaaclab_v1/source/tutorials/03_envs/register_rl_env_gym.html#registering-an-environment）
# 这个装饰器是将main函数进行包装，并加载对应配置文件解析为环境和代理的配置对象。
@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # # 加载文件中的代理的配置对象
    # log_dir = '/home/ppp/IsaacLab-1.4.1/logs/sb3/Isaac-Cartpole-v0/2025-03-08_15-37-31'
    # # env_cfg_dict = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    # # if isinstance(env_cfg, ManagerBasedRLEnvCfg):
    # #     env_cfg = ManagerBasedRLEnvCfg(**env_cfg_dict)
    # # elif isinstance(env_cfg, DirectRLEnvCfg):
    # #     env_cfg = DirectRLEnvCfg(**env_cfg_dict)
    # # elif isinstance(env_cfg, DirectMARLEnvCfg):
    # #     env_cfg = DirectMARLEnvCfg(**env_cfg_dict)
    # agent_cfg = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    # 将简单的YAML类型转换为Stable-Baselines类/组件。
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    # 将多个智能体的环境转换为单个智能体的环境，以便与Stable-Baselines3兼容
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # 通过gym的包装器来启用视频录制，并传入相应的参数
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    # configure the logger
    # 配置日志记录器
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    # When using multiple environments, each call to env.step() will effectively correspond to n_envs steps. 
    # To account for that, you can use save_freq = max(save_freq // n_envs, 1)
    # 当有多个环境时，每个对env.step()的调用实际上都将对应于n_envs步。因此此时n个step，实际上是n*n_envs个step。
    # 所以save_freq=1000时，如果有多个环境，那么实际上是1000*n_envs个step。要保持还是1000step保存异常就要减小save_freq。
    # 为了应对这个问题，可以使用save_freq = max(save_freq // n_envs, 1)
    # 保存的文件路径是log/model_{step_num}.zip，其中step_num是当前训练的步数。
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    # 指定保存的文件路径名（会自动添加.zip后缀），最后保存的文件路径为：log_dir/model.zip
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
