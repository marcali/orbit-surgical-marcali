# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
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

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import orbit.surgical.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    log_performance_dir = os.path.join(log_root_path, agent_cfg.load_run)
    log_performance_dir = os.path.abspath(log_performance_dir)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_performance_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Logging rewards

    log_performance_path = os.path.join(log_performance_dir, "performance_log.csv")
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    def initialize_csv(path):
        if not os.path.exists(path):
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Step", "Reward"])

    # Function to log the reward after each step
    def log_reward(step, reward,dones, extra, path):
        # Convert the tensor to a list
        reward_list = reward.tolist()
        dones_list = dones.tolist()
        
        # Flatten the list if it contains nested lists
        #flat_reward_list = [item for sublist in reward_list for item in sublist] if isinstance(reward_list[0], list) else reward_list
        flat_dones_list = [item for sublist in dones_list for item in sublist] if isinstance(dones_list[0], list) else dones_list
        #print(f"Step: {step}, Reward: {flat_reward_list}")
        print(f"Step: {step}, Extra: {extra}")
        print(f"Step: {step}, Dones: {flat_dones_list}")
        # Write the step and flattened reward list to the CSV file
        # with open(path, "a", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([step] + flat_reward_list)

    # Initialize the CSV file
    initialize_csv(log_performance_path)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # write git state to logs
    # ppo_runner.add_git_repo_to_log(log_performance_path)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rew, dones, extra = env.step(actions)
            timestep += 1
            log_reward(timestep, rew, dones, extra, log_performance_path)
            if args_cli.video:
                timestep += 1
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