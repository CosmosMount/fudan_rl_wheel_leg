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
        stiffness = {"f0": 6.0, "f1": 6.0, "wheel": 0.0}
        damping = {"f0": 0.5, "f1": 0.5, "wheel": 0.2}


    class asset(LeggedRobotCfg.asset):
        # file = "/root/gpufree-data/114514/wheel_leg/Wheel-Legged-Gym-master/resources/robots/RM/classis-URDF/urdf/classis-URDF.urdf"
        # file = "/root/gpufree-data/114514/wheel_leg/Wheel-Legged-Gym-master/resources/robots/wl/urdf/wl.urdf"
        # file = "/root/gpufree-data/wheel_leg/wheel_leg/Wheel-Legged-Gym-master/resources/robots/rm/urdf/rm_yly.urdf"
        # file = "/root/gpufree-data/wheel_leg/wheel_leg/Wheel-Legged-Gym-master/resources/robots/yly_v5/urdf/yly_v5.urdf"
        # file = "/root/gpufree-data/wheel_leg/wheel_leg/Wheel-Legged-Gym-master/resources/robots/infantry_V1/urdf/infantry_V1.urdf"
        # file = "/root/gpufree-data/final_version/plane/wheel_leg/Wheel-Legged-Gym-master/resources/robots/infantry_V1_old/urdf/old7.urdf"   原本的

        file = "/data/RL/RL/jump/wheel_leg/Wheel-Legged-Gym-master/resources/robots/infantry_V4/urdf/infantry_V4_increase.urdf"
        # file = "/root/gpufree-data/wheel_leg/wheel_leg/Wheel-Legged-Gym-master/resources/robots/old11/urdf/old11.urdf"

        name = "WheelLegged"
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged"
        max_iterations = 50000
