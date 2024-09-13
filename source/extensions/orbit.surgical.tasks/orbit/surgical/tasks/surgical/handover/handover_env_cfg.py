# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
    """Configuration for the handover scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_1_frame: FrameTransformerCfg = MISSING
    ee_2_frame: FrameTransformerCfg = MISSING
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

    # ee_1_pose: mdp.UniformPoseCommandCfg = MISSING

    # ee_2_pose: mdp.UniformPoseCommandCfg = MISSING

    object_pose_1 = mdp.UniformPoseCommandCfg(
        asset_name="robot_1",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
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

    object_pose_2 = mdp.UniformPoseCommandCfg(
        asset_name="robot_2",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
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

    # arm_1_action: ActionTerm = MISSING
    # gripper_1_action: ActionTerm | None = None

    # arm_2_action: ActionTerm = MISSING
    # gripper_2_action: ActionTerm | None = None

    # will be set by agent env cfg
    body_1_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_1_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING

    body_2_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_2_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_1_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_1")},
        )
        joint_1_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_1")},
        )
        joint_2_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_2")},
        )
        joint_2_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_2")},
        )
        # pose_1_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_1_pose"})
        # pose_2_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_2_pose"})
        object_position_1 = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot_1")}
        )
        object_position_2 = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot_2")}
        )
        target_object_position_1 = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose_1"})
        target_object_position_2 = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose_2"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_robot_1_joints: EventTerm = MISSING

    # reset_robot_2_joints: EventTerm = MISSING

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    # end_effector_1_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING), "command_name": "ee_1_pose"},
    # )
    # end_effector_1_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING), "command_name": "ee_1_pose"},
    # )

    # end_effector_2_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING), "command_name": "ee_2_pose"},
    # )
    # end_effector_2_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING), "command_name": "ee_2_pose"},
    # )

    # reaching_object = RewTerm(
    #     func=mdp.object_ee_distance, params={"std": 0.1, "ee_frame_cfg": SceneEntityCfg("ee_1_frame")}, weight=1.0
    # )
    # reaching_object = RewTerm(
    #     func=mdp.object_ee_distance, params={"std": 0.1, "ee_frame_cfg": SceneEntityCfg("ee_2_frame")}, weight=1.0
    # )

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.02,
            "command_name_receiving": "object_pose_1",
            "command_name_testing": "object_pose_2",
            "robot_receiving_cfg": SceneEntityCfg("robot_1"),
            "robot_testing_cfg": SceneEntityCfg("robot_2"),
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.02,
            "command_name_receiving": "object_pose_2",
            "command_name_testing": "object_pose_1",
            "robot_receiving_cfg": SceneEntityCfg("robot_2"),
            "robot_testing_cfg": SceneEntityCfg("robot_1"),
        },
        weight=5.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.02,
            "command_name_receiving": "object_pose_1",
            "command_name_testing": "object_pose_2",
            "robot_receiving_cfg": SceneEntityCfg("robot_1"),
            "robot_testing_cfg": SceneEntityCfg("robot_2"),
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.02,
            "command_name_receiving": "object_pose_2",
            "command_name_testing": "object_pose_1",
            "robot_receiving_cfg": SceneEntityCfg("robot_2"),
            "robot_testing_cfg": SceneEntityCfg("robot_1"),
        },
        weight=5.0,
    )

    # moving further arm from initial position penalty
    further_arm_deviation_robot_1 = RewTerm(
        func=mdp.second_arm_deviation_l1,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot_1",
                joint_names=[
                    "psm_yaw_joint",
                    "psm_pitch_end_joint",
                    "psm_main_insertion_joint",
                    "psm_tool_roll_joint",
                    "psm_tool_pitch_joint",
                    "psm_tool_yaw_joint",
                ],
            ),
            "command_name_receiving": "object_pose_1",
            "command_name_testing": "object_pose_2",
            "robot_receiving_cfg": SceneEntityCfg("robot_1"),
            "robot_testing_cfg": SceneEntityCfg("robot_2"),
        },
    )

    further_arm_deviation_robot_2 = RewTerm(
        func=mdp.second_arm_deviation_l1,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot_2",
                joint_names=[
                    "psm_yaw_joint",
                    "psm_pitch_end_joint",
                    "psm_main_insertion_joint",
                    "psm_tool_roll_joint",
                    "psm_tool_pitch_joint",
                    "psm_tool_yaw_joint",
                ],
            ),
            "command_name_receiving": "object_pose_2",
            "command_name_testing": "object_pose_1",
            "robot_receiving_cfg": SceneEntityCfg("robot_2"),
            "robot_testing_cfg": SceneEntityCfg("robot_1"),
        },
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)

    # penalty for object dropping
    object_drop = RewTerm(func=mdp.object_velocity, weight=-1.0)

    # joint velosity penalty
    joint_1_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot_1")},
    )
    joint_2_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot_2")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_1 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_1_vel", "weight": -1e-1, "num_steps": 10000}
    )
    # //:TODO: Check if this is correct, maybe need to modify sooner
    joint_vel_2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_2_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class HandoverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

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
        self.episode_length_s = 3.0
        # simulation settings
        self.sim.dt = 1.0 / 200.0
        self.viewer.eye = (0.2, 0.2, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.04)