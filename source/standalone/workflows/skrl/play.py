# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import csv
import pandas as pd

from omni.isaac.lab.utils.dict import print_dict

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg

import orbit.surgical.tasks  # noqa: F401


def main():
    """Play with skrl agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # create isaac environment
    #env = gym.make(args_cli.task, cfg=env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaaclab")`

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html
    models = {}
    # non-shared models
    if experiment_cfg["models"]["separate"]:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            clip_actions=True,
            clip_log_std=True,
            initial_log_std=-20.0,
            min_log_std=-20.0,
            max_log_std=-20.0,
            **process_skrl_cfg(experiment_cfg["models"]["policy"]),
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["value"]),
        )
    # shared models
    else:
        models["policy"] = shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["policy"]),
                process_skrl_cfg(experiment_cfg["models"]["value"]),
            ],
        )
        models["value"] = models["policy"]

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent = PPO(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Logging rewards
    run_num = "2"
    model_path = os.path.abspath(args_cli.checkpoint)
    log_performance_dir = os.path.dirname(os.path.join(log_root_path, model_path))
    log_performance_path = os.path.join(log_performance_dir, "performance_log.csv")
    print(f"[INFO] Logging RESULTS in directory: {log_root_path}")

    def initialize_csv(path):
        if not os.path.exists(path):
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Run","Step", "Reward", "Dones"])
    

    def log_reward(step, reward, term, time_out, extra, path, run_num):
        # Convert tensors to scalars or lists
        def convert_value(value):
            if isinstance(value, torch.Tensor):
                return value.item() if value.numel() == 1 else value.tolist()
            return value

        # Flatten the extra dictionary
        def flatten_dict(d, parent_key='', sep='/'):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = convert_value(v)
            return items

    # Prepare the data
        log_data = {
            'Run': run_num,
            'Step': step,
            'Reward': convert_value(reward),
            'Dones': convert_value(term)
        }

        if extra:
            extra_flat = flatten_dict(extra)
            log_data.update(extra_flat)

        # Convert log_data to DataFrame
        df_new = pd.DataFrame([log_data])

        # Check if the CSV file exists
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df_existing = pd.read_csv(path)
            # Combine the existing and new DataFrames, aligning columns
            df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
            # Fill NaN values with empty strings or zeros as appropriate
            df_combined = df_combined.fillna('')
        else:
            df_combined = df_new

        # Write the combined DataFrame back to CSV
        df_combined.to_csv(path, index=False)

    
    initialize_csv(log_performance_path)

    # initialize agent
    agent.init()
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # reset environment
    run_num = "2"
    timestep = 0
    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            # env stepping
            timestep += 1
            obs, rew, term, time_out, extra = env.step(actions)
            log_reward(timestep, rew, term, time_out, extra, log_performance_path, run_num)
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
