#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional
import time
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import torch
from pynput import keyboard


MUJOCO_ROOT = Path(__file__).resolve().parents[1]


def resource_path(relative_path: str) -> str:
    """Return a resource path relative to the portable MuJoCo project root."""
    return str(MUJOCO_ROOT / relative_path)

# ===== Runtime command state =====
running = True
cmd_lin_vel = 0.0
cmd_lin_vel_target = 0.0
cmd_yaw_vel = 0.0
cmd_yaw_vel_target = 0.0
DEFAULT_CMD_HEIGHT = 0.10
HEIGHT_MIN = 0.14
HEIGHT_MAX = 0.25


Jump_after_height_big = False
Jump_after_height = 0.21
FFF = 1.0
XXX = 0.1
YYY = 0.4
JUMP_POLICY_DURATION_S = 0.55
taumax = 30.0

cmd_height = DEFAULT_CMD_HEIGHT
pressed_keys = set()
requested_policy_name: Optional[str] = None
requested_policy_return_name: Optional[str] = None
requested_policy_duration_s: Optional[float] = None
mode_keys_down = set()
torque_enabled = True

# ===== Training-aligned constants =====
DOF_NAMES = [
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
]

LF0_IDX = 0
LF1_IDX = 1
LW_IDX = 2
RF0_IDX = 3
RF1_IDX = 4
RW_IDX = 5

# JUMP_XIAOFUDU_ONNX_PATH = "./actor/yuntai/tiao_2000.onnx"
# JUMP_XIAOFUDU_ONNX_PATH = "./actor/yuntai/猛烈跳跃软软的会收腿了但是不够高.onnx"
JUMP_XIAOFUDU_ONNX_PATH = resource_path("actor/yuntai/p60.50.2把urdf中力矩限制为50.onnx")

RECOVER_2500_ONNX_PATH = resource_path("actor/yuntai/上台阶3_angz+.onnx")
# RECOVER_2500_ONNX_PATH = "./actor/yuntai/6010.2加入随机地形terrain_proportions0.50.20.10.10.10.0.onnx"

# "E:\sim2sim_Isaacgym2mujuco\actor\yuntai\"
JUMP_XIAOFUDU_DEFAULT_DOF_POS = np.array([0.2, 0.4, 0.0, -0.2, -0.4, 0.0], dtype=np.float32)
RECOVER_2500_DEFAULT_DOF_POS = np.array([-0.23, -0.65, 0.0, 0.23, 0.65, 0.0], dtype=np.float32)
# RECOVER_2500_DEFAULT_DOF_POS = np.array([0.2, 0.4, 0.0, -0.2, -0.4, 0.0], dtype=np.float32)
OBS_DOF_POS_IDXS = np.array([0, 1, 3, 4], dtype=np.int64)

RECOVER_POLICY_NAME = "plane"
JUMP_POLICY_NAME = "jump"

SPACE_MODE_KEY = "__space__"
DEFAULT_XML_PATH = resource_path(
    "assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml"
)
DEFAULT_POLICY_NAME = RECOVER_POLICY_NAME
jump_policy_duration_s = JUMP_POLICY_DURATION_S

RUNTIME_PRESETS = {
    "jump": {
        "xml_path": DEFAULT_XML_PATH,
        "onnx_path": JUMP_XIAOFUDU_ONNX_PATH,
        "default_dof_pos": JUMP_XIAOFUDU_DEFAULT_DOF_POS,
        "command_scale": np.array([3.0, 0.25, 5.0], dtype=np.float32),
        # "p_gains": np.array([20.0, 20.0, 0.0, 20.0, 20.0, 0.0], dtype=np.float32),
        # "d_gains": np.array([1.0, 1.0, 0.2, 1.0, 1.0, 0.2], dtype=np.float32),
        "p_gains": np.array([6.0, 6.0, 0.0, 6.0, 6.0, 0.0], dtype=np.float32),
        "d_gains": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.2], dtype=np.float32),
    },
    "plane": {
        "xml_path": DEFAULT_XML_PATH,
        "onnx_path": RECOVER_2500_ONNX_PATH,
        "default_dof_pos": RECOVER_2500_DEFAULT_DOF_POS,
        "command_scale": np.array([3.0, 0.25, 5.0], dtype=np.float32),
        # "p_gains": np.array([20.0, 20.0, 0.0, 20.0, 20.0, 0.0], dtype=np.float32),
        # "d_gains": np.array([1.0, 1.0, 0.2, 1.0, 1.0, 0.2], dtype=np.float32),
        "p_gains": np.array([15.0, 15.0, 0.0, 15.0, 15.0, 0.0], dtype=np.float32),
        "d_gains": np.array([1.0, 1.0, 0.1, 1.0, 1.0, 0.1], dtype=np.float32),
    },
}

OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05

POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0
TORQUE_SCALE = 1.0

CLIP_ACTIONS = 100.0
CLIP_OBSERVATIONS = 100.0
HISTORY_LEN = 5
SERIAL_LEG_OFFSET = 0.0
SERIAL_LEG_L1 = 0.175
SERIAL_LEG_L2 = 0.208
SERIAL_MAP_EPS = 1e-4

CMD_LIN_VEL_STEP = 2.4
CMD_LIN_VEL_RAMP_RATE = 2.3
CMD_YAW_VEL_STEP = 4.0
CMD_YAW_VEL_RAMP_RATE = 4.0
CMD_HEIGHT_STEP = 0.01

LIN_VEL_MIN = -2.8
LIN_VEL_MAX = 2.8
YAW_VEL_MIN = -6.0
YAW_VEL_MAX = 6.0

SIDE_CAM_AZIMUTH_OFFSET = 90.0
SIDE_CAM_ELEVATION = -12.0
SIDE_CAM_DISTANCE = 1.6
SIDE_CAM_LOOKAT_HEIGHT = 0.2


def _clip_height(v: float) -> float:
    return float(np.clip(v, HEIGHT_MIN, HEIGHT_MAX))


def parse_bool_flag(value: str | None) -> bool:
    if value is None:
        return True
    return value.lower() in ("1", "true", "yes", "on")


def _refresh_cmd_from_keys():
    global cmd_lin_vel, cmd_lin_vel_target, cmd_yaw_vel, cmd_yaw_vel_target
    key_1 = "1" in pressed_keys
    key_2 = "2" in pressed_keys
    key_3 = "3" in pressed_keys
    key_4 = "4" in pressed_keys

    if key_1 and not key_2:
        cmd_lin_vel_target = CMD_LIN_VEL_STEP
    elif key_2 and not key_1:
        cmd_lin_vel_target = -CMD_LIN_VEL_STEP
    else:
        cmd_lin_vel_target = 0.0
        cmd_lin_vel = 0.0

    if key_3 and not key_4:
        cmd_yaw_vel_target = CMD_YAW_VEL_STEP
    elif key_4 and not key_3:
        cmd_yaw_vel_target = -CMD_YAW_VEL_STEP
    else:
        cmd_yaw_vel_target = 0.0
        cmd_yaw_vel = 0.0


def _update_cmd_lin_vel(dt: float):
    global cmd_lin_vel
    max_delta = CMD_LIN_VEL_RAMP_RATE * max(0.0, float(dt))
    delta = cmd_lin_vel_target - cmd_lin_vel
    if abs(delta) <= max_delta:
        cmd_lin_vel = cmd_lin_vel_target
    else:
        cmd_lin_vel += math.copysign(max_delta, delta)
    cmd_lin_vel = float(np.clip(cmd_lin_vel, LIN_VEL_MIN, LIN_VEL_MAX))


def _update_cmd_yaw_vel(dt: float):
    global cmd_yaw_vel
    cmd_yaw_vel = float(np.clip(cmd_yaw_vel_target, YAW_VEL_MIN, YAW_VEL_MAX))


def _key_name(key) -> Optional[str]:
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        return char.lower()
    vk = getattr(key, "vk", None)
    if isinstance(vk, int):
        if 48 <= vk <= 57:
            return chr(vk)
        if 96 <= vk <= 105:
            return str(vk - 96)
        if vk == 66:
            return "b"
    return None


def on_press(key):
    global running, cmd_height, torque_enabled
    global requested_policy_name, requested_policy_return_name, requested_policy_duration_s
    if key == keyboard.Key.esc:
        running = False
        return False

    if key == keyboard.Key.space:
        if SPACE_MODE_KEY in mode_keys_down:
            return
        mode_keys_down.add(SPACE_MODE_KEY)
        requested_policy_name = JUMP_POLICY_NAME
        requested_policy_return_name = RECOVER_POLICY_NAME if jump_policy_duration_s > 0.0 else None
        requested_policy_duration_s = jump_policy_duration_s if jump_policy_duration_s > 0.0 else None
        return

    k = _key_name(key)
    if k is None:
        return

    if k in ("1", "2", "3", "4"):
        pressed_keys.add(k)
        _refresh_cmd_from_keys()
        return
    if k in ("5", "6"):
        cmd_height = _clip_height(HEIGHT_MAX if k == "5" else HEIGHT_MIN)
        return
    if k == "9":
        mode_keys_down.discard(SPACE_MODE_KEY)
        requested_policy_name = RECOVER_POLICY_NAME
        requested_policy_return_name = None
        requested_policy_duration_s = None
        cmd_height = _clip_height(DEFAULT_CMD_HEIGHT)
        print("[POLICY] request -> plane")
        return
    if k == "b":
        torque_enabled = not torque_enabled
        print(f"[TORQUE] {'ON' if torque_enabled else 'OFF'}")
        return


def on_release(key):
    if key == keyboard.Key.space:
        mode_keys_down.discard(SPACE_MODE_KEY)
        return
    k = _key_name(key)
    if k is None:
        return
    if k in ("1", "2", "3", "4"):
        pressed_keys.discard(k)
        _refresh_cmd_from_keys()


def mj_sensor_id(m: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)


def find_actuator_for_joint(m: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"Joint '{joint_name}' not found.")
    for a in range(m.nu):
        if int(m.actuator_trnid[a, 0]) == jid:
            return a
    raise KeyError(f"No actuator found for joint '{joint_name}'.")


def clip_by_ctrlrange(u: np.ndarray, ctrl_lo: np.ndarray, ctrl_hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(u, ctrl_lo), ctrl_hi)


def clip_taumax(u: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=np.float32), -taumax, taumax).astype(np.float32)


def fit_vector(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = min(dim, flat.size)
    if n > 0:
        out[:n] = flat[:n]
    return out


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def quat_conj_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]], dtype=np.float32)


def quat_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q_xyzw[:3]
    qw = q_xyzw[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_rotate_inverse_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate_xyzw(quat_conj_xyzw(q_xyzw), v)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def serial_leg_polar(theta1: float, theta2: float, l1: float, l2: float, offset: float) -> np.ndarray:
    end_x = offset + l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    end_y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    l0 = math.hypot(end_x, end_y)
    theta0 = math.atan2(end_y, end_x) - math.pi / 2
    return np.array([l0, theta0], dtype=np.float32)


def serial_leg_jacobian(theta1: float, theta2: float, l1: float, l2: float, offset: float, eps: float) -> np.ndarray:
    plus_theta1 = serial_leg_polar(theta1 + eps, theta2, l1, l2, offset)
    minus_theta1 = serial_leg_polar(theta1 - eps, theta2, l1, l2, offset)
    plus_theta2 = serial_leg_polar(theta1, theta2 + eps, l1, l2, offset)
    minus_theta2 = serial_leg_polar(theta1, theta2 - eps, l1, l2, offset)

    dtheta1 = plus_theta1 - minus_theta1
    dtheta2 = plus_theta2 - minus_theta2
    dtheta1[1] = wrap_to_pi(dtheta1[1])
    dtheta2[1] = wrap_to_pi(dtheta2[1])
    jac = np.stack((dtheta1, dtheta2), axis=1) / (2.0 * eps)
    if abs(float(np.linalg.det(jac))) < 1e-6:
        raise ValueError(f"serial leg map near singular, theta1={theta1:.6f}, theta2={theta2:.6f}")
    return jac.astype(np.float32)


def serial_torque_to_ftp(tau_theta: np.ndarray, theta1: float, theta2: float) -> tuple[np.ndarray, np.ndarray]:
    jac = serial_leg_jacobian(
        theta1,
        theta2,
        SERIAL_LEG_L1,
        SERIAL_LEG_L2,
        SERIAL_LEG_OFFSET,
        SERIAL_MAP_EPS,
    )
    ftp = np.linalg.solve(jac.T, np.asarray(tau_theta, dtype=np.float32).reshape(2))
    tau_theta_back = jac.T @ ftp
    return ftp.astype(np.float32), tau_theta_back.astype(np.float32)


def scale_serial_leg_f(tau: np.ndarray, q: np.ndarray, f_scale: float) -> np.ndarray:
    if abs(float(f_scale) - 1.0) < 1e-6:
        return tau

    out = np.asarray(tau, dtype=np.float32).copy()
    left_theta1 = float(q[LF0_IDX])
    left_theta2 = float(q[LF1_IDX] + math.pi / 2)
    right_theta1 = float(-q[RF0_IDX])
    right_theta2 = float(-q[RF1_IDX] + math.pi / 2)

    left_tau_theta = np.array([out[LF0_IDX], out[LF1_IDX]], dtype=np.float32)
    right_tau_theta = np.array([-out[RF0_IDX], -out[RF1_IDX]], dtype=np.float32)
    left_ftp, _ = serial_torque_to_ftp(left_tau_theta, left_theta1, left_theta2)
    right_ftp, _ = serial_torque_to_ftp(right_tau_theta, right_theta1, right_theta2)

    left_ftp[0] *= float(f_scale)
    right_ftp[0] *= float(f_scale)
    left_jac = serial_leg_jacobian(left_theta1, left_theta2, SERIAL_LEG_L1, SERIAL_LEG_L2, SERIAL_LEG_OFFSET, SERIAL_MAP_EPS)
    right_jac = serial_leg_jacobian(right_theta1, right_theta2, SERIAL_LEG_L1, SERIAL_LEG_L2, SERIAL_LEG_OFFSET, SERIAL_MAP_EPS)
    left_tau_back = left_jac.T @ left_ftp
    right_tau_back = right_jac.T @ right_ftp

    out[LF0_IDX] = left_tau_back[0]
    out[LF1_IDX] = left_tau_back[1]
    out[RF0_IDX] = -right_tau_back[0]
    out[RF1_IDX] = -right_tau_back[1]
    return out.astype(np.float32)


def quat_wxyz_to_rpy(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = float(np.dot(q, q))
    if n < 1e-12:
        return np.zeros((3,), dtype=np.float64)
    q /= math.sqrt(n)
    w, x, y, z = q.tolist()
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def parse_onnx_io(sess: ort.InferenceSession):
    inputs = list(sess.get_inputs())
    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")
    obs_input = None
    hist_input = None
    for inp in inputs:
        name_l = inp.name.lower()
        if "history" in name_l:
            hist_input = inp
        elif obs_input is None:
            obs_input = inp
    if obs_input is None:
        obs_input = inputs[0]
    if hist_input is None and len(inputs) >= 2:
        hist_input = inputs[1]
    return obs_input.name, (hist_input.name if hist_input is not None else None)


def load_onnx_session(onnx_path: str, providers: list[str]) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def main():
    global cmd_height, jump_policy_duration_s
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=RUNTIME_PRESETS[DEFAULT_POLICY_NAME]["xml_path"])
    parser.add_argument("--onnx", type=str, default=RUNTIME_PRESETS[DEFAULT_POLICY_NAME]["onnx_path"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy_dt", type=float, default=0.01)
    parser.add_argument("--jump_onnx", type=str, default=JUMP_XIAOFUDU_ONNX_PATH)
    parser.add_argument("--jump_policy_duration", type=float, default=JUMP_POLICY_DURATION_S)
    parser.add_argument("--base_body_name", type=str, default="base_Link_del")
    parser.add_argument("--quat_source", type=str, default="xquat", choices=["xquat", "qpos", "sensor"])
    parser.add_argument("--angvel_in_body", type=int, default=1)
    parser.add_argument("--dof_vel_use_pos_diff", type=int, default=1)
    parser.add_argument("--torque_scale", type=float, default=TORQUE_SCALE)
    parser.add_argument("--jump_f_scale", type=float, default=FFF)
    parser.add_argument("--jump_f_scale_start", type=float, default=XXX)
    parser.add_argument("--jump_f_scale_end", type=float, default=YYY)
    parser.add_argument("--tag", nargs="?", const=True, default=Jump_after_height_big, type=parse_bool_flag)
    parser.add_argument("--jump_after_height", type=float, default=Jump_after_height)
    args = parser.parse_args()

    RUNTIME_PRESETS[JUMP_POLICY_NAME]["onnx_path"] = args.jump_onnx
    cmd_height = _clip_height(cmd_height)
    device = torch.device(args.device)

    # Single XML load only.
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    sess = load_onnx_session(args.onnx, providers)
    obs_name, hist_name = parse_onnx_io(sess)
    RUNTIME_PRESETS[RECOVER_POLICY_NAME]["onnx_path"] = args.onnx
    active_policy_name = RECOVER_POLICY_NAME
    active_p_gains = np.asarray(
        RUNTIME_PRESETS[active_policy_name]["p_gains"],
        dtype=np.float32,
    ).copy()
    active_d_gains = np.asarray(
        RUNTIME_PRESETS[active_policy_name]["d_gains"],
        dtype=np.float32,
    ).copy()
    active_default_dof_pos = np.asarray(
        RUNTIME_PRESETS[active_policy_name]["default_dof_pos"],
        dtype=np.float32,
    ).copy()
    active_default_obs_dof_pos = active_default_dof_pos[OBS_DOF_POS_IDXS]
    active_command_scale = np.asarray(
        RUNTIME_PRESETS[active_policy_name]["command_scale"],
        dtype=np.float32,
    ).copy()
    temporary_policy_return_name: Optional[str] = None
    temporary_policy_return_deadline: Optional[float] = None
    jump_f_scale_start_time: Optional[float] = None
    jump_f_scale_end_time: Optional[float] = None
    jump_policy_duration_s = max(0.0, float(args.jump_policy_duration))

    act_ids = np.array([find_actuator_for_joint(m, jn) for jn in DOF_NAMES], dtype=np.int32)
    ctrl_lo = np.asarray(m.actuator_ctrlrange[act_ids, 0], dtype=np.float32).copy()
    ctrl_hi = np.asarray(m.actuator_ctrlrange[act_ids, 1], dtype=np.float32).copy()
    unlimited = (np.abs(ctrl_lo) < 1e-9) & (np.abs(ctrl_hi) < 1e-9)
    ctrl_lo[unlimited], ctrl_hi[unlimited] = -1e6, 1e6

    base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.base_body_name)
    if base_bid < 0:
        raise RuntimeError(f"Body '{args.base_body_name}' not found.")
    if int(m.body_jntnum[base_bid]) <= 0:
        raise RuntimeError(f"Body '{args.base_body_name}' has no joint.")
    base_freejid = int(m.body_jntadr[base_bid])
    base_qpos_adr = int(m.jnt_qposadr[base_freejid])
    ori_sid = mj_sensor_id(m, "orientation")
    sensor_cache = {}

    def cache_sensor(name: str):
        sid = mj_sensor_id(m, name)
        if sid < 0:
            raise KeyError(f"Sensor '{name}' not found.")
        sensor_cache[name] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))

    def get_sensor(name: str) -> np.ndarray:
        adr, dim = sensor_cache[name]
        return np.asarray(d.sensordata[adr : adr + dim], dtype=np.float32).copy()

    cache_sensor("base_ang_vel")
    if ori_sid >= 0:
        cache_sensor("orientation")
    for dn in DOF_NAMES:
        cache_sensor(f"{dn}_p")
        cache_sensor(f"{dn}_v")

    sim_dt = 0.005
    steps_per_policy = max(1, int(round(args.policy_dt / sim_dt)))
    eff_policy_dt = steps_per_policy * sim_dt

    history = None
    last_actions = np.zeros((6,), dtype=np.float32)
    vel_last_pos: Optional[np.ndarray] = None
    vel_last_time: Optional[float] = None
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def get_base_quat_wxyz() -> np.ndarray:
        if args.quat_source == "xquat":
            return np.asarray(d.xquat[base_bid], dtype=np.float32).copy()
        if args.quat_source == "qpos":
            return np.asarray(d.qpos[base_qpos_adr + 3 : base_qpos_adr + 7], dtype=np.float32).copy()
        if ori_sid >= 0:
            return get_sensor("orientation").reshape(-1)
        return np.asarray(d.xquat[base_bid], dtype=np.float32).copy()

    def initialize_side_camera(viewer):
        base_pos = np.asarray(d.xpos[base_bid], dtype=np.float64).copy()
        yaw_deg = math.degrees(quat_wxyz_to_rpy(get_base_quat_wxyz())[2])
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = base_pos
        viewer.cam.lookat[2] += SIDE_CAM_LOOKAT_HEIGHT
        viewer.cam.distance = SIDE_CAM_DISTANCE
        viewer.cam.elevation = SIDE_CAM_ELEVATION
        viewer.cam.azimuth = yaw_deg + SIDE_CAM_AZIMUTH_OFFSET

    def read_dof_pos() -> np.ndarray:
        q = np.zeros((6,), dtype=np.float32)
        for i, dn in enumerate(DOF_NAMES):
            q[i] = float(get_sensor(f"{dn}_p")[0])
        return q

    def read_dof_vel_sensor() -> np.ndarray:
        qd = np.zeros((6,), dtype=np.float32)
        for i, dn in enumerate(DOF_NAMES):
            qd[i] = float(get_sensor(f"{dn}_v")[0])
        return qd

    def estimate_vel(q: np.ndarray) -> np.ndarray:
        nonlocal vel_last_pos, vel_last_time
        if vel_last_pos is None or vel_last_time is None:
            out = np.zeros_like(q)
        else:
            dt = float(d.time) - vel_last_time
            if dt > 1e-12:
                diff = np.remainder(q - vel_last_pos + math.pi, 2.0 * math.pi) - math.pi
                out = diff / dt
            else:
                out = np.zeros_like(q)
        vel_last_pos = q.copy()
        vel_last_time = float(d.time)
        return out

    def build_state():
        q_wxyz = get_base_quat_wxyz()
        q_xyzw = wxyz_to_xyzw(q_wxyz)
        base_ang = get_sensor("base_ang_vel").reshape(-1)
        if args.angvel_in_body == 0:
            base_ang = quat_rotate_inverse_xyzw(q_xyzw, base_ang)
        q = read_dof_pos()
        qd = estimate_vel(q) if int(args.dof_vel_use_pos_diff) != 0 else read_dof_vel_sensor()
        q_obs = q[OBS_DOF_POS_IDXS].copy()
        projected_g = quat_rotate_inverse_xyzw(q_xyzw, g_world)
        return q_wxyz, base_ang, projected_g, q, q_obs, qd

    def build_obs(base_ang: np.ndarray, projected_g: np.ndarray, q_obs: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cmd_raw = np.array([cmd_lin_vel, cmd_yaw_vel, cmd_height], dtype=np.float32)
        cmd_raw[0] = float(np.clip(cmd_raw[0], LIN_VEL_MIN, LIN_VEL_MAX))
        cmd_raw[1] = float(np.clip(cmd_raw[1], YAW_VEL_MIN, YAW_VEL_MAX))
        cmd_raw[2] = float(np.clip(cmd_raw[2], HEIGHT_MIN, HEIGHT_MAX))
        obs_raw = np.concatenate(
            [
                base_ang * OBS_SCALE_ANG_VEL,
                projected_g,
                cmd_raw * active_command_scale,
                (q_obs - active_default_obs_dof_pos) * OBS_SCALE_DOF_POS,
                qd * OBS_SCALE_DOF_VEL,
                last_actions,
            ],
            axis=0,
        ).astype(np.float32)
        obs = np.clip(obs_raw, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS).astype(np.float32)
        return obs, obs_raw

    def compute_torque(actions: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        act = np.clip(fit_vector(actions, 6), -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        pos_ref = act * POS_ACTION_SCALE
        pos_ref[2] = 0.0
        pos_ref[5] = 0.0
        vel_ref = act * VEL_ACTION_SCALE
        vel_ref[0] = 0.0
        vel_ref[1] = 0.0
        vel_ref[3] = 0.0
        vel_ref[4] = 0.0
        tau = (
            active_p_gains * (pos_ref + active_default_dof_pos - q)
            + active_d_gains * (vel_ref - qd)
        ) * float(args.torque_scale)
        if (
            jump_f_scale_start_time is not None
            and jump_f_scale_end_time is not None
            and jump_f_scale_start_time <= float(d.time) < jump_f_scale_end_time
        ):
            tau = scale_serial_leg_f(tau, q, float(args.jump_f_scale))
        return clip_taumax(tau)

    def switch_policy(policy_name: str) -> None:
        nonlocal sess, obs_name, hist_name, history, last_actions
        nonlocal active_policy_name
        nonlocal active_p_gains, active_d_gains, active_default_dof_pos, active_default_obs_dof_pos
        nonlocal active_command_scale
        preset = RUNTIME_PRESETS[policy_name]
        onnx_path = str(preset["onnx_path"])
        sess = load_onnx_session(onnx_path, providers)
        obs_name, hist_name = parse_onnx_io(sess)
        active_command_scale = np.asarray(preset["command_scale"], dtype=np.float32).copy()
        active_p_gains = np.asarray(preset["p_gains"], dtype=np.float32).copy()
        active_d_gains = np.asarray(preset["d_gains"], dtype=np.float32).copy()
        active_default_dof_pos = np.asarray(preset["default_dof_pos"], dtype=np.float32).copy()
        active_default_obs_dof_pos = active_default_dof_pos[OBS_DOF_POS_IDXS]
        active_policy_name = policy_name
        history = None
        last_actions[:] = 0.0
        print(f"[POLICY] switch -> {active_policy_name}: {onnx_path}")

    def handle_policy_requests() -> None:
        global requested_policy_name, requested_policy_return_name, requested_policy_duration_s
        nonlocal temporary_policy_return_name, temporary_policy_return_deadline
        nonlocal jump_f_scale_start_time, jump_f_scale_end_time
        if requested_policy_name is None:
            return

        pending_policy_name = requested_policy_name
        pending_policy_return_name = requested_policy_return_name
        pending_policy_duration_s = requested_policy_duration_s

        requested_policy_name = None
        requested_policy_return_name = None
        requested_policy_duration_s = None
        temporary_policy_return_name = None
        temporary_policy_return_deadline = None
        jump_f_scale_start_time = None
        jump_f_scale_end_time = None

        if pending_policy_name != active_policy_name:
            switch_policy(pending_policy_name)

        if (
            active_policy_name == pending_policy_name
            and pending_policy_return_name is not None
            and pending_policy_duration_s is not None
            and pending_policy_duration_s > 0.0
        ):
            temporary_policy_return_name = pending_policy_return_name
            temporary_policy_return_deadline = float(d.time) + float(pending_policy_duration_s)
            if pending_policy_name == JUMP_POLICY_NAME:
                phase_start = float(np.clip(args.jump_f_scale_start, 0.0, 1.0))
                phase_end = float(np.clip(args.jump_f_scale_end, 0.0, 1.0))
                if phase_end > phase_start:
                    jump_f_scale_start_time = float(d.time) + float(pending_policy_duration_s) * phase_start
                    jump_f_scale_end_time = float(d.time) + float(pending_policy_duration_s) * phase_end

    def handle_temporary_policy_return() -> None:
        global cmd_height
        nonlocal temporary_policy_return_name, temporary_policy_return_deadline
        nonlocal jump_f_scale_start_time, jump_f_scale_end_time
        if (
            temporary_policy_return_name is None
            or temporary_policy_return_deadline is None
            or float(d.time) < temporary_policy_return_deadline
        ):
            return

        return_policy_name = temporary_policy_return_name
        temporary_policy_return_name = None
        temporary_policy_return_deadline = None
        jump_f_scale_start_time = None
        jump_f_scale_end_time = None

        if return_policy_name != active_policy_name:
            switch_policy(return_policy_name)
        if bool(args.tag):
            cmd_height = _clip_height(float(args.jump_after_height))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    i = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:

        initialize_side_camera(viewer)

        while viewer.is_running() and running:
            i = i + 1
            handle_policy_requests()
            handle_temporary_policy_return()
            _update_cmd_lin_vel(eff_policy_dt)
            _update_cmd_yaw_vel(eff_policy_dt)

            _, base_ang, projected_g, q, q_obs, qd = build_state()
            print(base_ang)

            obs, obs_hist = build_obs(base_ang, projected_g, q_obs, qd)
            if history is None:
                history = np.repeat(obs_hist.reshape(1, -1), HISTORY_LEN, axis=0)
            else:
                history[:-1, :] = history[1:, :]
                history[-1, :] = obs_hist

            feed = {obs_name: obs.reshape(1, -1).astype(np.float32)}
            if hist_name is not None:
                feed[hist_name] = history.reshape(1, -1).astype(np.float32)
            action_raw = np.asarray(sess.run(None, feed)[0], dtype=np.float32).reshape(-1)
            actions = np.clip(fit_vector(action_raw, 6), -CLIP_ACTIONS, CLIP_ACTIONS)
            last_actions[:] = actions.copy()

            q_step = q
            qd_step = qd
            for step_idx in range(steps_per_policy):
                # time.sleep(sim_dt)
                tau = compute_torque(actions, q_step, qd_step)
                u = clip_taumax(clip_by_ctrlrange(tau, ctrl_lo, ctrl_hi))
                d.ctrl[:] = 0.0
                if torque_enabled:
                    d.ctrl[act_ids] = u
                # d.ctrl[act_ids[LW_IDX]] = +2.0
                # d.ctrl[act_ids[RW_IDX]] = +2.0

                mujoco.mj_step(m, d)
                if step_idx < steps_per_policy - 1:
                    _, _, _, q_step, _, qd_step = build_state()
            viewer.sync()

    try:
        listener.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
