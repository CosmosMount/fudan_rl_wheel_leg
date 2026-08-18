# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class WheelLeggedCfg(LeggedRobotCfg):

    # class domain_rand(LeggedRobotCfg.domain_rand):
    #     randomize_friction = True
    #     friction_range = [0.25, 1.6]
    #     randomize_restitution = True
    #     restitution_range = [0.0, 0.6]
    #     randomize_base_mass = True
    #     added_mass_range = [-1.5, 2.0]
    #     randomize_inertia = True
    #     randomize_inertia_range = [0.85, 1.15]
    #     randomize_base_com = True
    #     rand_com_vec = [0.04, 0.04, 0.04]
    #     push_robots = True
    #     push_interval_s = 6
    #     max_push_vel_xy = 1.8
    #     randomize_Kp = True
    #     randomize_Kp_range = [0.92, 1.08]
    #     randomize_Kd = True
    #     randomize_Kd_range = [0.92, 1.08]
    #     randomize_motor_torque = True
    #     randomize_motor_torque_range = [0.92, 1.08]
    #     randomize_default_dof_pos = True
    #     randomize_default_dof_pos_range = [-0.04, 0.04]
    #     randomize_action_delay = True
    #     delay_ms_range = [0, 8]
    #     randomize_leg_kp = True
    #     leg_kp_range = [0.92, 1.08]
    #     randomize_leg_kd = True
    #     leg_kd_range = [0.92, 1.08]
    #     randomize_leg_motor_torque = True
    #     leg_motor_torque_range = [0.92, 1.08]
    #     randomize_leg_joint_offset = True
    #     leg_joint_offset_range = [-0.04, 0.04]
    #     randomize_wheel_motor_torque = True
    #     wheel_motor_torque_range = [0.92, 1.08]
    #     randomize_wheel_action_delay = True
    #     wheel_delay_ms_range = [0, 6]
    #     randomize_wheel_action_deadzone = True
    #     wheel_action_deadzone_range = [0.0, 0.04]
    #     randomize_wheel_radius_scale = True
    #     wheel_radius_scale_range = [0.97, 1.03]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.05, 2.5]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-3.0, 4.0]
        randomize_inertia = True
        randomize_inertia_range = [0.7, 1.3]
        randomize_base_com = True
        rand_com_vec = [0.07, 0.07, 0.06]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 3.2
        randomize_Kp = True
        randomize_Kp_range = [0.85, 1.15]
        randomize_Kd = True
        randomize_Kd_range = [0.85, 1.15]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.85, 1.15]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.07, 0.07]
        randomize_action_delay = True
        delay_ms_range = [0, 14]
        randomize_leg_kp = True
        leg_kp_range = [0.85, 1.15]
        randomize_leg_kd = True
        leg_kd_range = [0.85, 1.15]
        randomize_leg_motor_torque = True
        leg_motor_torque_range = [0.85, 1.15]
        randomize_leg_joint_offset = True
        leg_joint_offset_range = [-0.07, 0.07]
        randomize_wheel_motor_torque = True
        wheel_motor_torque_range = [0.85, 1.15]
        randomize_wheel_action_delay = True
        wheel_delay_ms_range = [0, 10]
        randomize_wheel_action_deadzone = True
        wheel_action_deadzone_range = [0.0, 0.08]
        randomize_wheel_radius_scale = True
        wheel_radius_scale_range = [0.93, 1.07]

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.1]  # x,y,z [m]
        # default_joint_angles = { "lf0_Joint": -0.23, 
        #                         "lf1_Joint": -0.65, 
        #                         "l_wheel_Joint": 0.0, 
        #                         "rf0_Joint": 0.23, 
        #                         "rf1_Joint": 0.65, 
        #                         "r_wheel_Joint": 0.0, 
        #                         }
        pos = [0.0, 0.0, 0.2]  # x,y,z [m]
        default_joint_angles = { "lf0_Joint": 0.2, 
                                "lf1_Joint": 0.4, 
                                "l_wheel_Joint": 0.0, 
                                "rf0_Joint": -0.2, 
                                "rf1_Joint": -0.4, 
                                "r_wheel_Joint": 0.0, 
                                }


    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"f0": 15.0, "f1": 15.0, "wheel": 0.0}
        damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.2}


    class asset(LeggedRobotCfg.asset):


        # file = "/root/gpufree-data/final_version/plane/resources/robots/infantry_V1/urdf/infantry_V1.urdf"
        file = "/root/gpufree-data/final_version/plane/resources/robots/infantry_V4/urdf/infantry_V4_increase.urdf"



        name = "WheelLegged"
        offset = 0.0
        # l1 = 0.215
        # l2 = 0.258 旧车的
        l1 = 0.175
        l2 = 0.208
        penalize_contacts_on =        ["rf", "lf", "base"]
        terminate_after_contacts_on = ["rf", "lf", "base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged"
        max_iterations = 50000
