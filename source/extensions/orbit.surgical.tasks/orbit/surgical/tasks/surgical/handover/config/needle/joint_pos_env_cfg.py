# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from orbit.surgical.tasks.surgical.handover import mdp
from orbit.surgical.tasks.surgical.handover.handover_env_cfg import HandoverEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_CFG  # isort: skip


@configclass
class NeedleHandoverEnvCfg(HandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        self.scene.robot_1 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.1, 0.0, 0.15)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (-0.1, 0.0, 0.15)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Set actions for the specific robot type (PSM)
        self.actions.body_1_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.4,
            use_default_offset=True,
        )
        self.actions.body_2_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.4,
            use_default_offset=True,
        )
        self.actions.finger_1_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
        )
        self.actions.finger_2_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
        )
        # Set the body name for the end effector
        self.commands.object_pose_1.body_name = "psm_tool_tip_link"
        self.commands.object_pose_2.body_name = "psm_tool_tip_link"

        # Set Suture Needle as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/tube/tube_02.usd",
        #         scale=(0.02, 0.04, 0.02),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=8,
        #             max_angular_velocity=200,
        #             max_linear_velocity=200,
        #             max_depenetration_velocity=1.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )
        
        #Set an obstacle
        # self.scene.obstacle = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.0, 0.05), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(1.0, 1.0, 1.0),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=16,
        #             max_angular_velocity=0.1,
        #             max_linear_velocity=0.1,
        #             max_depenetration_velocity=1.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class NeedleHandoverEnvCfg_PLAY(NeedleHandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False