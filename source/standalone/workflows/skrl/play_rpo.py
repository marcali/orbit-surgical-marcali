# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl using the RPO algorithm.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import csv

# Import RPO instead of PPO
from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg

import orbit.surgical.tasks  # noqa: F401


def main():
    """Play with skrl RPO agent."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # Create Isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaaclab")`

    # Instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html
    models = {}
    # Non-shared models
    if experiment_cfg["models"]["separate"]:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["policy"]),
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["value"]),
        )
    # Shared models
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

    # Configure and instantiate RPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/rpo.html
    agent_cfg = RPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # Avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["write_interval"] = 0  # Don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # Don't generate checkpoints

    agent = RPO(
        models=models,
        memory=None,  # Memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # Specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # Logging rewards
    run_num = "1"
    model_path = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else ""
    log_performance_dir = os.path.join(log_root_path, "inference_logs")
    os.makedirs(log_performance_dir, exist_ok=True)
    log_performance_filename = f"performance_log_run_{run_num}.csv"
    log_performance_path = os.path.join(log_performance_dir, log_performance_filename)
    print(f"[INFO] Logging RESULTS in directory: {log_performance_path}")

    def initialize_csv(path):
        if not os.path.exists(path):
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Step", "Reward"])

    def log_reward(step, reward, term, time_out, extra, path, run_num):

        # Convert tensors to scalars or lists
        if isinstance(reward, torch.Tensor):
            reward_value = reward.item() if reward.numel() == 1 else reward.tolist()
        else:
            reward_value = reward

        if isinstance(term, torch.Tensor):
            dones_list = term.tolist()
        else:
            dones_list = term

        # Flatten the list if it contains nested lists
        flat_dones_list = (
            [item for sublist in dones_list for item in sublist] if isinstance(dones_list[0], list) else dones_list
        )
        dones_value = flat_dones_list[0] if len(flat_dones_list) == 1 else flat_dones_list

        # Create a dict to hold all data
        log_data = {}
        log_data["Run"] = run_num  # Include run number
        log_data["Step"] = step
        log_data["Reward"] = reward_value
        log_data["Dones"] = dones_value

        # Extract extra['log'] items
        if "log" in extra:
            for key, value in extra["log"].items():
                # Convert tensors to scalars or lists
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.tolist()
                log_data[key] = value

        # If the CSV file doesn't exist or is empty, write the headers
        file_exists = os.path.isfile(path)
        write_headers = not file_exists or os.path.getsize(path) == 0

        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_headers:
                # Write headers
                headers = list(log_data.keys())
                writer.writerow(headers)
            # Write data
            row = [log_data[key] for key in log_data.keys()]
            writer.writerow(row)

    initialize_csv(log_performance_path)
    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # Initialize agent
    agent.init()
    agent.load(resume_path)
    # Set agent to evaluation mode
    agent.set_running_mode("eval")

    # Reset environment
    run_num = "run 1"
    timestep = 0
    obs, _ = env.reset()
    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Agent stepping
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            # Env stepping
            timestep += 1
            obs, rew, term, time_out, extra = env.step(actions)
            log_reward(timestep, rew, term, time_out, extra, log_performance_path, run_num)
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation app
    simulation_app.close()
