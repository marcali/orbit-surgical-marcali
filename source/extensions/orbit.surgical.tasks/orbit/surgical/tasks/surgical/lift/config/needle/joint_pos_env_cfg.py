# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, DeformableObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass

from orbit.surgical.tasks.surgical.lift import mdp
from orbit.surgical.tasks.surgical.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_CFG  # isort: skip


@configclass
class NeedleLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        self.scene.robot = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (PSM)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
                        # scale={
            #     "psm_yaw_joint": 0.5,
            #     "psm_pitch_end_joint": 0.2,
            #     "psm_main_insertion_joint": 0.3,
            #     "psm_tool_roll_joint": 0.1,
            #     "psm_tool_pitch_joint": 0.1,
            #     "psm_tool_yaw_joint": 0.5,
            # },
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "psm_tool_tip_link"

        # # Set Suture Needle as object
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

        # self.scene.object = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/tube/tube_02.usd",
        #         scale=(0.2, 0.2, 0.2),
        #         rigid_props=DeformableBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             max_depenetration_velocity=1.0,
        #             contact_offset=0.001,
        #             rest_offset=0.0,
        #         ),
        #     )
        # )

                # Rigid Object cone obstacle
        # self.scene.obstacle = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.0, 0.015), rot=(1, 0, 0, 0)),
        #     # contact_pose=torch.tensor([-1.0, 0.0, 0.0, 1, 0, 0, 0]),
        #     # non_contact_pose=torch.tensor([-1.0, 0.0, 1.0, 1, 0, 0, 0]),
        #     spawn=sim_utils.ConeCfg(
        #         radius=0.05,
        #         height=0.1,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             disable_gravity=False,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=True,
        #             #contact_offset=0.05,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        #         #activate_contact_sensors=True,
        #     ),
        # )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


        # override rewards
        #self.rewards.grasp_needle.params["open_joint_pos1"] = 0.05
        #self.rewards.grasp_needle.params["open_joint_pos2"] = -0.05


@configclass
class NeedleLiftEnvCfg_PLAY(NeedleLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
