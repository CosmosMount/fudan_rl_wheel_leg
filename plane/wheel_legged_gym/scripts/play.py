# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np

import isaacgym
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry

try:
    from pynput import keyboard
except ImportError:
    print("Missing dependency: pynput. Please install it with: pip install pynput")
    raise


# --------------------
# Global command state
# --------------------
cmd_x = 0.0
ang_vel = 0.0
cmd_height = 0.2
running = True
turn_left_pressed = False
turn_right_pressed = False

LIN_VEL_CMD = 2.0
YAW_STEP = 2.0
HEIGHT_STEP = 0.02

# Initial viewer camera. Applied once after the environments are created.
INITIAL_CAMERA_POSITION = [20.0, -20.0, 10.0]
INITIAL_CAMERA_LOOK_AT = [20.0, 40.0, 0.0]



def update_yaw_cmd():
    global ang_vel
    if turn_left_pressed and not turn_right_pressed:
        ang_vel = YAW_STEP
    elif turn_right_pressed and not turn_left_pressed:
        ang_vel = -YAW_STEP
    else:
        ang_vel = 0.0


def on_press(key):
    global cmd_x, ang_vel, cmd_height, running
    global turn_left_pressed, turn_right_pressed

    if key == keyboard.Key.esc:
        running = False
        print("[CMD] quit (ESC)")
        return False

    try:
        k = key.char.lower()
    except Exception:
        return

    if k == "q":
        running = False
        print("[CMD] quit (q)")
        return False
    if k == "w":
        cmd_x = LIN_VEL_CMD
        print(f"[CMD] forward: x={cmd_x:.2f}")
    elif k == "s":
        cmd_x = -LIN_VEL_CMD
        print(f"[CMD] backward: x={cmd_x:.2f}")
    elif k == "a":
        if not turn_left_pressed:
            print("[CMD] turn left (hold)")
        turn_left_pressed = True
        update_yaw_cmd()
    elif k == "d":
        if not turn_right_pressed:
            print("[CMD] turn right (hold)")
        turn_right_pressed = True
        update_yaw_cmd()
    elif k == "e":
        cmd_x = 0.0
        turn_left_pressed = False
        turn_right_pressed = False
        update_yaw_cmd()
        print("[CMD] stop")
    elif k == "x":
        cmd_height += HEIGHT_STEP
        print(f"[CMD] height up: h={cmd_height:.2f}")
    elif k == "c":
        cmd_height -= HEIGHT_STEP
        print(f"[CMD] height down: h={cmd_height:.2f}")


def on_release(key):
    global turn_left_pressed, turn_right_pressed
    try:
        k = key.char.lower()
    except Exception:
        return

    if k == "a":
        turn_left_pressed = False
        update_yaw_cmd()
    elif k == "d":
        turn_right_pressed = False
        update_yaw_cmd()
    return


def apply_manual_commands(env, env_cfg):
    global cmd_x, ang_vel, cmd_height

    cmd_x = float(
        np.clip(
            cmd_x,
            env_cfg.commands.ranges.lin_vel_x[0],
            env_cfg.commands.ranges.lin_vel_x[1],
        )
    )
    ang_vel = float(
        np.clip(
            ang_vel,
            env_cfg.commands.ranges.ang_vel_yaw[0],
            env_cfg.commands.ranges.ang_vel_yaw[1],
        )
    )
    cmd_height = float(
        np.clip(
            cmd_height,
            env_cfg.commands.ranges.height[0],
            env_cfg.commands.ranges.height[1],
        )
    )

    env.commands[:, 2] = cmd_height

    jump_ids = getattr(env, "jump_ramp_idx", None)
    if jump_ids is None or len(jump_ids) == 0:
        env.commands[:, 0] = cmd_x
        env.commands[:, 1] = ang_vel
        return

    manual_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    manual_mask[jump_ids] = False
    manual_ids = manual_mask.nonzero(as_tuple=False).flatten()
    if len(manual_ids) != 0:
        env.commands[manual_ids, 0] = cmd_x
        env.commands[manual_ids, 1] = ang_vel

    env.commands[jump_ids, 0] = env_cfg.commands.jump_ramp_lin_vel_x
    env.commands[jump_ids, 2] = env_cfg.commands.jump_ramp_height
    env.commands[jump_ids, 3] = env_cfg.commands.jump_ramp_heading


def play(args):
    global running

    print("\n====== Keyboard Control Mode (NO Enter) ======")
    print("w      : forward")
    print("s      : backward")
    print("a      : hold to turn left")
    print("d      : hold to turn right")
    print("e      : stop")
    print("x      : height up")
    print("c      : height down")
    print("q/ESC  : quit")
    print("camera : fixed overview")
    print("=============================================\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.episode_length_s = 20
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.lift_robots = False
    env_cfg.domain_rand.downward_impulse_robots = False
    env_cfg.domain_rand.downward_impulse_interval_s = 3
    env_cfg.domain_rand.downward_impulse_vel_range = [2.4, 2.8]
    # env_cfg.domain_rand.vmc_force_events = True
    env_cfg.terrain.curriculum = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if getattr(env, "viewer", None) is not None:
        env.set_camera(INITIAL_CAMERA_POSITION, INITIAL_CAMERA_LOOK_AT)

    apply_manual_commands(env, env_cfg)
    obs, obs_history = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    is_sequence_policy = bool(ppo_runner.alg.actor_critic.is_sequence)

    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy to:", path)

    i = 0
    try:
        while running and i < 100000:
            apply_manual_commands(env, env_cfg)
            if is_sequence_policy:
                actions, _ = policy(obs, obs_history)
            else:
                actions = policy(obs)

            obs, _, _, _, _, obs_history = env.step(actions)
            apply_manual_commands(env, env_cfg)

            if i % 50 == 0:
                vz = env.root_states[0, 9].item()
                yaw_rate = env.base_ang_vel[0, 2].item()
                # left_F = env.vmc_F[0, 0].item()
                # right_F = env.vmc_F[0, 1].item()
                print(
                    f"[{i}] vz={vz:.3f}, cmd_x={env.commands[0, 0].item():.2f}, "
                    f"cmd_yaw={env.commands[0, 1].item():.3f}, real_yaw={yaw_rate:.3f}, "
                    # f"F_left={left_F:.2f}, F_right={right_F:.2f}"
                )
            i += 1
    finally:
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    EXPORT_POLICY = False
    args = get_args()
    play(args)
