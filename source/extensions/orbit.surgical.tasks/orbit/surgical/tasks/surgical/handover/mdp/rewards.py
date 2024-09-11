# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking of the position error using L2-norm.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame). The position error is computed as the L2-norm
#     of the difference between the desired and current positions.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
#     # distance reward
#     return torch.norm(curr_pos_w - des_pos_w, dim=1)


# def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking orientation error using shortest path.

#     The function computes the orientation error between the desired orientation (from the command) and the
#     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
#     path between the desired and current orientations.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current orientations
#     des_quat_b = command[:, 3:7]
#     des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
#     curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
#     # Computes the rotation difference between two quaternions.
#     return quat_error_magnitude(curr_quat_w, des_quat_w)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # rewarded if the object is lifted above the threshold 0.02
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name_receiving: str,
    command_name_testing: str,
    robot_receiving_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_testing_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot_receiving: RigidObject = env.scene[robot_receiving_cfg.name]
    robot_testing: RigidObject = env.scene[robot_testing_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command_receiving = env.command_manager.get_command(command_name_receiving)
    command_testing = env.command_manager.get_command(command_name_testing)

    # compute the desired position in the world frame for the receiving robot
    des_pos_b_receiving = command_receiving[:, :3]
    des_pos_w_receiving, _ = combine_frame_transforms(
        robot_receiving.data.root_state_w[:, :3], robot_receiving.data.root_state_w[:, 3:7], des_pos_b_receiving
    )
    # distance of the end-effector to the object: (num_envs,)
    distance_receiving = torch.norm(des_pos_w_receiving - object.data.root_pos_w[:, :3], dim=1)

    # compute the desired position in the world frame for the testing robot
    des_pos_b_testing = command_testing[:, :3]
    des_pos_w_testing, _ = combine_frame_transforms(
        robot_testing.data.root_state_w[:, :3], robot_testing.data.root_state_w[:, 3:7], des_pos_b_testing
    )
    # distance of the end-effector to the object: (num_envs,)
    distance_testing = torch.norm(des_pos_w_testing - object.data.root_pos_w[:, :3], dim=1)
    # //:TODO: check how it performs with the original code using std
    # Choose the closest robot
    closest_robot_distance = torch.min(distance_receiving, distance_testing)

    # Compute the reward for the closest robot
    dist_closest = torch.where(closest_robot_distance == 0, 0.001, closest_robot_distance)
    #reward = 1 - torch.tanh(dist_closest / std)
    reward = (object.data.root_pos_w[:, 2] > minimal_height) * (torch.div(1, torch.square(10 * dist_closest)))

    return reward


def second_arm_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name_receiving: str,
    command_name_testing: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_receiving_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_testing_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    robot_receiving: RigidObject = env.scene[robot_receiving_cfg.name]
    robot_testing: RigidObject = env.scene[robot_testing_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command_receiving = env.command_manager.get_command(command_name_receiving)
    command_testing = env.command_manager.get_command(command_name_testing)

    # compute the desired position in the world frame for the receiving robot
    des_pos_b_receiving = command_receiving[:, :3]
    des_pos_w_receiving, _ = combine_frame_transforms(
        robot_receiving.data.root_state_w[:, :3], robot_receiving.data.root_state_w[:, 3:7], des_pos_b_receiving
    )
    # distance of the end-effector to the object: (num_envs,)
    distance_receiving = torch.norm(des_pos_w_receiving - object.data.root_pos_w[:, :3], dim=1)

    # compute the desired position in the world frame for the testing robot
    des_pos_b_testing = command_testing[:, :3]
    des_pos_w_testing, _ = combine_frame_transforms(
        robot_testing.data.root_state_w[:, :3], robot_testing.data.root_state_w[:, 3:7], des_pos_b_testing
    )
    # distance of the end-effector to the object: (num_envs,)
    distance_testing = torch.norm(des_pos_w_testing - object.data.root_pos_w[:, :3], dim=1)

    # Penalize or reward based on distance comparison
    sign = torch.where(distance_receiving < distance_testing, 0.0, -1.0)

    # Compute joint deviation from the default position for the receiving robot
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Apply the sign to the sum of absolute deviations
    deviation = torch.sum(torch.abs(angle), dim=1)
    return deviation * sign


def object_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_lin_vel_w = object.data.root_lin_vel_w.clone()
    object_lin_vel_norm = torch.norm(object_lin_vel_w, dim=-1, p=2)
    penalty = torch.where(object_lin_vel_norm > 1, 1, 0)
    return penalty