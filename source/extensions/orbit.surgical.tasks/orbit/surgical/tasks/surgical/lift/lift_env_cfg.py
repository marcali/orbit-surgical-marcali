# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING
        # obstacle: will be populated by agent env cfg
    #obstacle: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(1.0, 1.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.12),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # object position and velocity
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #object_velocity = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})

        # obstacle position and velocity
        #obstacle_position = ObsTerm(func=mdp.obstacle_position_in_robot_root_frame)
        #obstacle_velocity = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("obstacle")})

        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

        # domain randomization
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (0.7, 1.3),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 250,
    #     },
    # )
    # robot_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.95, 1.05),
    #         "operation": "scale",
    #     },
    # )

    # robot_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.95, 1.05),
    #         "operation": "scale",
    #     },
    # )

    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
    #         "damping_distribution_params": (0.75, 1.5),  # default: 0.1
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # -- object
    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (0.7, 1.3),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 250,
    #     },
    # )
    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.4, 1.6),
    #         "operation": "scale",
    #     },
    # )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # reset_obstacle_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("obstacle", body_names="Obstacle"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # previously weight 0.4
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # increased lifing reward
    # best result: no reach, lift weight 15, dt 0.01, 5.sec
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # sd 0.3, best weight 16.0
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    # sd 0.05, best weight 5.0
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    #best weight -1e-3
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    # penalized agent for taking large actions. encourages to take small controlled actions
    #action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)

    #best weight -1e-2
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #best weight -2.0
    object_drop = RewTerm(func=mdp.object_velocity, weight=-2.0)

    #best weight -0.01
    # joint_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["psm_tool_pitch_joint", "psm_tool_roll_joint"])},
    # )
    
    #best weight -0.01
    applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})

    #collision = RewTerm(func=mdp.rewards.object_Collision, params={}, weight=-0.5)
    
    # grasp_needle = RewTerm(
    #     func=mdp.grasp_needle,
    #     weight=5.0,
    #     params={
    #         "threshold": 0.01,
    #         "open_joint_pos1": MISSING,
    #         "open_joint_pos2": MISSING,
    #         "asset_cfg1": SceneEntityCfg("robot", joint_names=["psm_tool_gripper1_joint"]),
    #         "asset_cfg2": SceneEntityCfg("robot", joint_names=["psm_tool_gripper2_joint"]),
    #     },
    # )
    
    # collision penalty

    # shelf_collision = RewTerm(func=mdp.collision_penalty, params={}, weight=-0.2)
    # object_collision = RewTerm(func=mdp.dynamic_penalty, params={"std": 0.3}, weight=-0.2)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )

    #best weight -1.0
    action_rate2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1.0, "num_steps": 15000}
    )

    #collision1 = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "collision", "weight": -8.0, "num_steps": 15000})

    # grasp_needle = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "grasp_needle", "weight": 17, "num_steps": 25000}
    # )
    #collision2 = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "collision", "weight": -10.0, "num_steps": 000})


    #best weight -1.0
    joint_vel2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1.0, "num_steps": 15000}
    )

    #best weight -0.1
    torque_limits = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "applied_torque_limits", "weight": -0.1, "num_steps": 15000})
    #does nothing for rsl rl and kind of impreves for slrl
    # object_moving = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_drop", "weight": -3, "num_steps": 20000}
    # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 2.0
        # simulation settings
        self.sim.dt = 1.0 / 200.0
        self.viewer.eye = (0.2, 0.2, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.04)
