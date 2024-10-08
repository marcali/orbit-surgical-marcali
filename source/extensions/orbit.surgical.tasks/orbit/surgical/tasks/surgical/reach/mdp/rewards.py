# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

class ObstacleAvoidancePenalty(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the obstacle avoidance penalty term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        obstacle_cfg = SceneEntityCfg("obstacle")

        self.ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self.obstacle: RigidObject = env.scene[obstacle_cfg.name]
        self.std = 0.3  # Standard deviation for smoothness in distance-based penalty
        self.obstacle_threshold = 0.01  # Distance threshold for proximity penalty
        self.collision_threshold = 0.005  # Threshold to detect collision movement

        # Access the entities from the scene
        self.ee: FrameTransformer = env.scene["ee_frame"]
        self.obstacle: RigidObject = env.scene["obstacle"]

        # Save the initial obstacle position for tracking
        self.initial_obst_pos = self.obstacle.data.root_pos_w.clone()

    def __call__(self, env: ManagerBasedRLEnv):
        """Compute the total penalty for the agent."""
        collision_penalty = self._collision_penalty(env)
        proximity_penalty = self._proximity_penalty(env)
        total_penalty = collision_penalty + proximity_penalty
        return total_penalty

    def _collision_penalty(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize the agent if the obstacle moves (indicating a collision)."""
        obstacle_velocity = self.obstacle.data.root_lin_vel_w
        obstacle_delta = self.obstacle.data.root_pos_w - self.initial_obst_pos
        # Check if the obstacle has moved beyond the collision threshold
        collision = torch.where(
            torch.norm(obstacle_delta, dim=-1) + torch.norm(obstacle_velocity, dim=-1) > self.collision_threshold,
            1.0,  # Apply maximum penalty for collision
            0.0   # No penalty if no collision
        )
        return collision

    def _proximity_penalty(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize the agent as it approaches the obstacle, with a smooth gradient."""
        # Compute the distance between the end-effector and the obstacle
        obstacle_pos = self.obstacle.data.root_pos_w
        ee_pos = self.ee.data.target_pos_w[..., 0, :]  # End-effector position

        distance_to_obstacle = torch.norm(obstacle_pos - ee_pos, dim=1)
        penalty = 1 - torch.tanh(distance_to_obstacle / self.obstacle_threshold)

        # Apply a penalty if the end-effector is within the threshold distance to the obstacle
        return torch.where(distance_to_obstacle < self.obstacle_threshold, penalty, 0.0)

