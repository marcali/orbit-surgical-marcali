# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train an RL agent with skrl using the RPO algorithm.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

#${IsaacLab_PATH}/isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Needle-PSM-v0  --headless --video

import argparse

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

# Import RPO instead of PPO
from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg

import orbit.surgical.tasks  # noqa: F401


def main():
    """Train with the skrl RPO agent."""
    # Read the seed from command line
    args_cli_seed = args_cli.seed

    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # Set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # Update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # Multi-GPU training config
    if args_cli.distributed:
        # Update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # Max iterations for training
    if args_cli.max_iterations:
        experiment_cfg["trainer"]["timesteps"] = args_cli.max_iterations * experiment_cfg["agent"]["rollouts"]

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # Create Isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # Wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaaclab")`

    # Set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

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

    # Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    # https://skrl.readthedocs.io/en/latest/api/memories/random.html
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Configure and instantiate the RPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/rpo.html
    agent_cfg = RPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # Avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent = RPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # Configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/api/trainers.html
    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # Train the agent
    trainer.train()

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation app
    simulation_app.close()
