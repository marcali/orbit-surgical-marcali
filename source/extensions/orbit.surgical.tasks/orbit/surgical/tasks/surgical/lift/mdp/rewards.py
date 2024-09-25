# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import csv
import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# Define log path
log_root_path = os.path.join("logs", "rsl_rl", "play")
log_root_path = os.path.abspath(log_root_path)
os.makedirs(log_root_path, exist_ok=True)


# Helper function to log data to CSV
def log_to_csv(file_path, data):
    # Assuming this function logs data to a CSV file
    with open(file_path, "w") as f:
        for item in data:
            f.write(f"{item}\n")


# lifting
def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    reward = torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    return reward


# reaching
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
    reward = 1 - torch.tanh(object_ee_distance / std)
    return reward

# minimizing robot object distance
def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    reward = (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
    return reward


def object_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_lin_vel_w = object.data.root_lin_vel_w
    object_lin_vel_norm = torch.norm(object_lin_vel_w, dim=-1, p=2)
    penalty = torch.where(object_lin_vel_norm > 1, 1, 0)
    return penalty


def align_ee_handle(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    object_quat = object.data.root_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(object_quat)

    # Get the batch size (number of environments)
    batch_size = ee_tcp_rot_mat.size(0)

    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    handle_x = handle_x.unsqueeze(1).expand(
        batch_size, 3, 1
    )  # Expand to [batch_size, 3, 1]
    handle_y = handle_y.unsqueeze(1).expand(batch_size, 3, 1)

    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]


    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y).squeeze(-1).squeeze(-1)
    reward = 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)
    return reward


def grasp_needle(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos1: float,
    open_joint_pos2: float,
    asset_cfg1: SceneEntityCfg,
    asset_cfg2: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos1 = env.scene[asset_cfg1.name].data.joint_pos[
        :, asset_cfg1.joint_ids
    ]
    gripper_joint_pos2 = env.scene[asset_cfg2.name].data.joint_pos[
        :, asset_cfg2.joint_ids
    ]

    distance = torch.norm(object_pos_w - ee_tcp_pos, dim=1)
    is_close = distance <= threshold

    # Compute reward for each gripper
    reward_gripper_1 = is_close * torch.sum(
        open_joint_pos1 - gripper_joint_pos1, dim=-1
    )
    # print("reward_gripper_1", reward_gripper_1)
    reward_gripper_2 = is_close * torch.sum(
        open_joint_pos2 - gripper_joint_pos2, dim=-1
    )
    # print("reward_gripper_2", reward_gripper_2)

    # Combine rewards
    total_reward = 10 * (reward_gripper_1 + reward_gripper_2)
    # multiplied by weight for logging
    # modified_reward = total_reward * 1
    # log_to_csv(os.path.join(log_root_path, "grasp_needle.csv"), modified_reward.tolist())

    return total_reward

############################### collision

class object_Collision(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("object")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        obstacle_cfg = SceneEntityCfg("obstacle")

        self.object: RigidObject = env.scene[object_cfg.name]
        self.ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self.obstacle: RigidObject = env.scene[obstacle_cfg.name]
        self.std = 0.3

        self.initial_obst_pos = self.obstacle.data.root_pos_w.clone()
    
    def __call__(self, env: ManagerBasedRLEnv):

        #collision = self.collision_penalty(env)
        proximity_penalty = self.dynamic_penalty(env)
        return proximity_penalty
    
    def collision_penalty(
        self,
        env: ManagerBasedRLEnv
    ) -> torch.Tensor:
        # Corrected to use the obstacle
        obst_vel = self.obstacle.data.root_lin_vel_w
        obst_delta = self.obstacle.data.root_pos_w - self.initial_obst_pos
        moved = torch.where(
            torch.norm(obst_delta, dim=-1, p=2) + torch.norm(obst_vel, dim=-1, p=2)
            > 0.005,
            1.0,
            0.0,
        )

        return moved

    def dynamic_penalty(
        self,
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
        """Compute the penalty for approaching the obstacle with a smooth gradient."""

        obstacle_pos_w = self.obstacle.data.root_pos_w  # Shape: (num_envs, 3)
        ee_w = self.ee.data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)

        # Compute Euclidean distance between end-effector and obstacle
        obstacle_ee_distance = torch.norm(obstacle_pos_w - ee_w, dim=1)  # Shape: (num_envs,)

        print("obstacle_ee_distance:", obstacle_ee_distance)

        # Define threshold for penalty
        obstacle_threshold = 0.1

        # Compute smooth penalty using tanh
        penalty_values = 1 - torch.tanh(obstacle_ee_distance / obstacle_threshold)  # Shape: (num_envs,)
        print("penalty_values (1 - tanh(distance / threshold)):", penalty_values)

        # Apply penalty: higher penalty as distance decreases below the threshold
        penalty = torch.where(
            obstacle_ee_distance < obstacle_threshold,
            penalty_values,  # Smooth penalty value
            0.0              # Scalar value; broadcasted to match shape
        )
        print("penalty:", penalty)

        return penalty