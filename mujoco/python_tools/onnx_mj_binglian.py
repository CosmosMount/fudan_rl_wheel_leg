#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import time
import argparse
import math
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
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


DEFAULT_XML_PATH = resource_path("assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml")

JUMP_ONNX_PATH = resource_path("actor/yuntai/p60.50.2把urdf中力矩限制为50.onnx")

DEFAULT_ONNX_PATH = resource_path("actor/yuntai/上台阶3_angz+.onnx")

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
SPACE_MODE_KEY = "__space__"
OBS_DOF_POS_IDXS = np.array([0, 1, 3, 4], dtype=np.int64)
PLANE_POLICY_NAME = "plane"
JUMP_POLICY_NAME = "jump"
DEFAULT_POLICY_NAME = PLANE_POLICY_NAME
JUMP_POLICY_DURATION_S = 0.55

FFF = 1.0
XXX = 0.1
YYY = 0.4
Jump_after_height_big = True
Jump_after_height = 0.26
DEFAULT_CMD_HEIGHT = 0.10
HEIGHT_MIN = 0.12
HEIGHT_MAX = 0.27
TAU_MAX = 30.0



PLANE_DEFAULT_DOF_POS = np.array([-0.23, -0.65, 0.0, 0.23, 0.65, 0.0], dtype=np.float32)
JUMP_DEFAULT_DOF_POS = np.array([0.2, 0.4, 0.0, -0.2, -0.4, 0.0], dtype=np.float32)
PLANE_P_GAINS = np.array([15.0, 15.0, 0.0, 15.0, 15.0, 0.0], dtype=np.float32)
PLANE_D_GAINS = np.array([1.0, 1.0, 0.1, 1.0, 1.0, 0.1], dtype=np.float32)
JUMP_P_GAINS = np.array([6.0, 6.0, 0.0, 6.0, 6.0, 0.0], dtype=np.float32)
JUMP_D_GAINS = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2], dtype=np.float32)
PLANE_COMMAND_SCALE = np.array([3.0, 0.25, 5.0], dtype=np.float32)
JUMP_COMMAND_SCALE = np.array([3.0, 0.25, 5.0], dtype=np.float32)
RUNTIME_PRESETS = {
    PLANE_POLICY_NAME: {
        "onnx_path": DEFAULT_ONNX_PATH,
        "default_dof_pos": PLANE_DEFAULT_DOF_POS,
        "p_gains": PLANE_P_GAINS,
        "d_gains": PLANE_D_GAINS,
        "command_scale": PLANE_COMMAND_SCALE,
    },
    JUMP_POLICY_NAME: {
        "onnx_path": JUMP_ONNX_PATH,
        "default_dof_pos": JUMP_DEFAULT_DOF_POS,
        "p_gains": JUMP_P_GAINS,
        "d_gains": JUMP_D_GAINS,
        "command_scale": JUMP_COMMAND_SCALE,
    },
}
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05

POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0
CLIP_ACTIONS = 100.0
CLIP_OBSERVATIONS = 100.0
HISTORY_LEN = 5

DEFAULT_OFFSET = 1.6614
MAP_EPS = 1e-6

CMD_LIN_VEL = 2.0
CMD_YAW_VEL = 4.0
CMD_RAMP_TIME = 0.5


LEFT_GAS_SPRING_ACTUATOR_NAME = "Left_loop1_motor"
RIGHT_GAS_SPRING_ACTUATOR_NAME = "Right_loop1_motor"
LEFT_GAS_SPRING_CTRL  = 300.0
RIGHT_GAS_SPRING_CTRL = 300.0
GAS_SPRING_FORCE = RIGHT_GAS_SPRING_CTRL * 1.23


def fit_vector(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = min(dim, flat.size)
    if n:
        out[:n] = flat[:n]
    return out


def parse_bool_flag(value: str | None) -> bool:
    if value is None:
        return True
    return value.lower() in ("1", "true", "yes", "on")


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


def solve_leg_geometry(phi1: float, phi4: float, l1: float, l2: float) -> tuple[float, float, float, float]:
    x_b = l1 * math.cos(phi1)
    y_b = l1 * math.sin(phi1)
    x_d = l1 * math.cos(phi4)
    y_d = l1 * math.sin(phi4)

    dx = x_d - x_b
    dy = y_d - y_b
    a0 = 2.0 * l2 * dx
    b0 = 2.0 * l2 * dy
    c0 = dx * dx + dy * dy
    disc = max(a0 * a0 + b0 * b0 - c0 * c0, 0.0)
    phi2 = 2.0 * math.atan2(b0 + math.sqrt(disc), a0 + c0)

    x_c = x_b + l2 * math.cos(phi2)
    y_c = y_b + l2 * math.sin(phi2)
    phi3 = math.atan2(y_c - y_d, x_c - x_d)
    l0 = math.hypot(x_c, y_c)
    phi0 = math.atan2(y_c, x_c)
    return phi2, phi3, phi0, l0


def solve_phi3_left(phi1: float, phi4: float, l1: float, l2: float) -> float:
    _, phi3, _, _ = solve_leg_geometry(phi1, phi4, l1, l2)
    return phi3 - phi4 - math.pi / 2


def solve_phi3_right(phi1: float, phi4: float, l1: float, l2: float) -> float:
    _, phi3, _, _ = solve_leg_geometry(phi1, phi4, l1, l2)
    return -phi3 + phi4 + math.pi / 2


def solve_phi_piandao(solver, phi1: float, phi4: float, l1: float, l2: float, eps: float) -> tuple[float, float]:
    dphi_dphi1 = (solver(phi1 + eps, phi4, l1, l2) - solver(phi1 - eps, phi4, l1, l2)) / (2.0 * eps)
    dphi_dphi4 = (solver(phi1, phi4 + eps, l1, l2) - solver(phi1, phi4 - eps, l1, l2)) / (2.0 * eps)
    return float(dphi_dphi1), float(dphi_dphi4)


def solve_jacobian_analytic(phi1: float, phi4: float, l1: float, l2: float) -> tuple[float, float]:
    _, phi3, _, l0 = solve_leg_geometry(phi1, phi4, l1, l2)
    phi_lf = abs(phi3 - phi4 - math.pi / 2.0)
    beta = 0.5 * abs(phi1 - phi4)
    denom = 2.0 * l2 * math.cos(beta - phi_lf)
    if abs(denom) < 1e-6:
        denom = math.copysign(1e-6, denom if denom != 0.0 else 1.0)
    k = (l1 * math.cos(beta) + l2 * math.cos(beta - phi_lf)) / denom
    # k = l0 / (2.0 * l2 * math.cos(abs(phi3 - phi4 - beta)))
    return float(k), float(-k)


def force_map(phi1: float, phi4: float, l1: float, l2: float) -> tuple[np.ndarray, np.ndarray, float]:
    phi2, phi3, phi0, l0 = solve_leg_geometry(phi1, phi4, l1, l2)
    j11 = math.sin(phi0 - phi3) * l1 * math.sin(phi1 - phi2) / math.sin(phi3 - phi2)
    j12 = math.cos(phi0 - phi3) * l1 * math.sin(phi1 - phi2) / (math.sin(phi3 - phi2) * l0)
    j21 = math.sin(phi0 - phi2) * l1 * math.sin(phi3 - phi4) / math.sin(phi3 - phi2)
    j22 = math.cos(phi0 - phi2) * l1 * math.sin(phi3 - phi4) / (math.sin(phi3 - phi2) * l0)
    matrix = np.array([[j11, j12], [j21, j22]], dtype=np.float32)
    det = j11 * j22 - j12 * j21
    if abs(det) < 1e-6:
        raise ValueError(f"leg force map near singular, det={det:.6e}")
    matrix_inv = np.array([[j22 / det, -j12 / det], [-j21 / det, j11 / det]], dtype=np.float32)
    return matrix, matrix_inv, l0


def mj_actuator_id(m: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise KeyError(f"Actuator '{name}' not found.")
    return int(actuator_id)


def mj_sensor_id(m: mujoco.MjModel, name: str) -> int:
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"Sensor '{name}' not found.")
    return int(sensor_id)


def clip_actuator_ctrl(m: mujoco.MjModel, act_id: int, value: float) -> float:
    lo = float(m.actuator_ctrlrange[act_id, 0])
    hi = float(m.actuator_ctrlrange[act_id, 1])
    if abs(lo) < 1e-9 and abs(hi) < 1e-9:
        return float(value)
    return float(np.clip(value, lo, hi))


def parse_onnx_io(sess: ort.InferenceSession) -> tuple[str, Optional[str]]:
    inputs = list(sess.get_inputs())
    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")
    hist_input = None
    for inp in inputs:
        if "history" in inp.name.lower():
            hist_input = inp
            break
    if hist_input is None and len(inputs) >= 2:
        hist_input = inputs[1]
    return inputs[0].name, hist_input.name if hist_input is not None else None


def load_onnx_session(onnx_path: str, device: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


@dataclass
class CommandState:
    running: bool = True
    cmd_lin_vel: float = 0.0
    cmd_yaw_vel: float = 0.0
    cmd_lin_vel_target: float = 0.0
    cmd_yaw_vel_target: float = 0.0
    cmd_height: float = DEFAULT_CMD_HEIGHT
    pressed_keys: set[str] = None
    mode_keys_down: set[str] = None
    requested_policy_name: Optional[str] = None
    requested_policy_return_name: Optional[str] = None
    requested_policy_duration_s: Optional[float] = None
    jump_policy_duration_s: float = JUMP_POLICY_DURATION_S
    torque_enabled: bool = True

    def __post_init__(self) -> None:
        if self.pressed_keys is None:
            self.pressed_keys = set()
        if self.mode_keys_down is None:
            self.mode_keys_down = set()

    def on_press(self, key: Any) -> Optional[bool]:
        if key == keyboard.Key.esc:
            self.running = False
            return False
        if key == keyboard.Key.space:
            if SPACE_MODE_KEY in self.mode_keys_down:
                return None
            self.mode_keys_down.add(SPACE_MODE_KEY)
            self.requested_policy_name = JUMP_POLICY_NAME
            self.requested_policy_return_name = PLANE_POLICY_NAME if self.jump_policy_duration_s > 0.0 else None
            self.requested_policy_duration_s = self.jump_policy_duration_s if self.jump_policy_duration_s > 0.0 else None
            return None
        name = self.key_name(key)
        if name is None:
            return None
        if name in {"1", "2", "3", "4"}:
            self.pressed_keys.add(name)
            self.refresh()
        elif name == "5":
            self.cmd_height = HEIGHT_MAX
            print(f"[CMD] height={self.cmd_height:.3f}")
        elif name == "6":
            self.cmd_height = HEIGHT_MIN
            print(f"[CMD] height={self.cmd_height:.3f}")
        elif name == "9":
            self.requested_policy_name = PLANE_POLICY_NAME
            self.requested_policy_return_name = None
            self.requested_policy_duration_s = None
            self.mode_keys_down.discard(SPACE_MODE_KEY)
            self.cmd_height = DEFAULT_CMD_HEIGHT
            print("[POLICY] request -> plane")
        elif name == "b":
            self.torque_enabled = not self.torque_enabled
            print(f"[TORQUE] {'ON' if self.torque_enabled else 'OFF'}")
        return None

    def on_release(self, key: Any) -> None:
        if key == keyboard.Key.space:
            self.mode_keys_down.discard(SPACE_MODE_KEY)
            return
        name = self.key_name(key)
        if name in {"1", "2", "3", "4"}:
            self.pressed_keys.discard(name)
            self.refresh()

    def refresh(self) -> None:
        forward = "1" in self.pressed_keys
        backward = "2" in self.pressed_keys
        left = "3" in self.pressed_keys
        right = "4" in self.pressed_keys
        self.cmd_lin_vel_target = CMD_LIN_VEL if forward and not backward else -CMD_LIN_VEL if backward and not forward else 0.0
        self.cmd_yaw_vel_target = CMD_YAW_VEL if left and not right else -CMD_YAW_VEL if right and not left else 0.0
        if not forward and not backward:
            self.cmd_lin_vel = 0.0
        if not left and not right:
            self.cmd_yaw_vel = 0.0

    def update_ramp(self, dt: float, ramp_time: float) -> None:
        self.cmd_lin_vel = self.ramp_value(
            self.cmd_lin_vel,
            self.cmd_lin_vel_target,
            CMD_LIN_VEL,
            dt,
            ramp_time,
        )
        self.cmd_yaw_vel = float(self.cmd_yaw_vel_target)

    @staticmethod
    def ramp_value(value: float, target: float, max_value: float, dt: float, ramp_time: float) -> float:
        if abs(target) <= 1e-9:
            return 0.0
        if ramp_time <= 1e-9:
            return float(target)
        max_delta = abs(float(max_value)) * max(0.0, float(dt)) / float(ramp_time)
        delta = float(target) - float(value)
        if abs(delta) <= max_delta:
            return float(target)
        return float(value) + math.copysign(max_delta, delta)

    @staticmethod
    def key_name(key: Any) -> Optional[str]:
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


@dataclass
class Handles:
    lf0: int
    lw: int
    rf0: int
    rw: int
    lf20: int
    rf20: int
    left_gas_spring: int
    right_gas_spring: int
    base_bid: int
    base_qpos_adr: int
    l20_qpos_adr: int
    r20_qpos_adr: int
    l20_qvel_adr: int
    r20_qvel_adr: int
    lf0_qpos_adr: int
    rf0_qpos_adr: int
    lf0_qvel_adr: int
    rf0_qvel_adr: int


class BinglianRuntime:
    def __init__(self, args: argparse.Namespace, command_state: CommandState) -> None:
        self.args = args
        self.command_state = command_state
        self.m = mujoco.MjModel.from_xml_path(args.xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

        self.handles = self.build_handles()
        self.sensor_cache: dict[str, tuple[int, int]] = {}
        self.init_sensor_cache()

        RUNTIME_PRESETS[PLANE_POLICY_NAME]["onnx_path"] = args.onnx
        self.active_policy_name = DEFAULT_POLICY_NAME
        self.active_policy_path = str(RUNTIME_PRESETS[self.active_policy_name]["onnx_path"])
        self.active_default_dof_pos = np.asarray(RUNTIME_PRESETS[self.active_policy_name]["default_dof_pos"], dtype=np.float32).copy()
        self.active_default_obs_dof_pos = self.active_default_dof_pos[OBS_DOF_POS_IDXS]
        self.active_p_gains = np.asarray(RUNTIME_PRESETS[self.active_policy_name]["p_gains"], dtype=np.float32).copy()
        self.active_d_gains = np.asarray(RUNTIME_PRESETS[self.active_policy_name]["d_gains"], dtype=np.float32).copy()
        self.active_command_scale = np.asarray(RUNTIME_PRESETS[self.active_policy_name]["command_scale"], dtype=np.float32).copy()
        self.temporary_policy_return_name: Optional[str] = None
        self.temporary_policy_return_deadline: Optional[float] = None
        self.jump_f_scale_start_time: Optional[float] = None
        self.jump_f_scale_end_time: Optional[float] = None

        self.sess = load_onnx_session(self.active_policy_path, args.device)
        self.obs_name, self.hist_name = parse_onnx_io(self.sess)
        self.history: Optional[np.ndarray] = None
        self.last_actions = np.zeros((6,), dtype=np.float32)

        self.sim_dt = float(args.sim_dt)
        self.steps_per_policy = max(1, int(round(args.policy_dt / self.sim_dt)))
        self.eff_policy_dt = self.steps_per_policy * self.sim_dt
        self.vel_last_pos: Optional[np.ndarray] = None
        self.vel_last_time: Optional[float] = None
        self.g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.debug_counter = 0
        self.jacobian_debug_counter = 0

        print(f"[XML] {args.xml}")
        print(f"[POLICY] {self.active_policy_name}: {self.active_policy_path}")
        print(f"[POLICY] jump: {RUNTIME_PRESETS[JUMP_POLICY_NAME]['onnx_path']} duration={self.command_state.jump_policy_duration_s:.3f}s")
        print(f"[CMD] ramp_time={args.cmd_ramp_time:.3f}s")
        print("[KEY] Space jump, 1 forward, 2 backward, 3 left, 4 right, 5 high, 6 low, Esc quit")

    def switch_policy(self, policy_name: str) -> None:
        preset = RUNTIME_PRESETS[policy_name]
        onnx_path = str(preset["onnx_path"])
        self.sess = load_onnx_session(onnx_path, self.args.device)
        self.obs_name, self.hist_name = parse_onnx_io(self.sess)
        self.active_policy_name = policy_name
        self.active_policy_path = onnx_path
        self.active_default_dof_pos = np.asarray(preset["default_dof_pos"], dtype=np.float32).copy()
        self.active_default_obs_dof_pos = self.active_default_dof_pos[OBS_DOF_POS_IDXS]
        self.active_p_gains = np.asarray(preset["p_gains"], dtype=np.float32).copy()
        self.active_d_gains = np.asarray(preset["d_gains"], dtype=np.float32).copy()
        self.active_command_scale = np.asarray(preset["command_scale"], dtype=np.float32).copy()
        self.history = None
        self.last_actions[:] = 0.0
        print(f"[POLICY] switch -> {self.active_policy_name}: {self.active_policy_path}")

    def handle_policy_requests(self) -> None:
        if self.command_state.requested_policy_name is None:
            return

        pending_policy_name = self.command_state.requested_policy_name
        pending_policy_return_name = self.command_state.requested_policy_return_name
        pending_policy_duration_s = self.command_state.requested_policy_duration_s

        self.command_state.requested_policy_name = None
        self.command_state.requested_policy_return_name = None
        self.command_state.requested_policy_duration_s = None
        self.temporary_policy_return_name = None
        self.temporary_policy_return_deadline = None
        self.jump_f_scale_start_time = None
        self.jump_f_scale_end_time = None

        if pending_policy_name != self.active_policy_name:
            self.switch_policy(pending_policy_name)

        if (
            self.active_policy_name == pending_policy_name
            and pending_policy_return_name is not None
            and pending_policy_duration_s is not None
            and pending_policy_duration_s > 0.0
        ):
            self.temporary_policy_return_name = pending_policy_return_name
            self.temporary_policy_return_deadline = float(self.d.time) + float(pending_policy_duration_s)
            if pending_policy_name == JUMP_POLICY_NAME:
                phase_start = float(np.clip(self.args.jump_f_scale_start, 0.0, 1.0))
                phase_end = float(np.clip(self.args.jump_f_scale_end, 0.0, 1.0))
                if phase_end > phase_start:
                    self.jump_f_scale_start_time = float(self.d.time) + float(pending_policy_duration_s) * phase_start
                    self.jump_f_scale_end_time = float(self.d.time) + float(pending_policy_duration_s) * phase_end

    def handle_temporary_policy_return(self) -> None:
        if (
            self.temporary_policy_return_name is None
            or self.temporary_policy_return_deadline is None
            or float(self.d.time) < self.temporary_policy_return_deadline
        ):
            return

        return_policy_name = self.temporary_policy_return_name
        self.temporary_policy_return_name = None
        self.temporary_policy_return_deadline = None
        self.jump_f_scale_start_time = None
        self.jump_f_scale_end_time = None

        if return_policy_name != self.active_policy_name:
            self.switch_policy(return_policy_name)
        if bool(self.args.tag):
            self.command_state.cmd_height = float(np.clip(self.args.jump_after_height, HEIGHT_MIN, HEIGHT_MAX))
            print(f"[CMD] jump return height={self.command_state.cmd_height:.3f}")

    def build_handles(self) -> Handles:
        base_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.args.base_body_name)
        if base_bid < 0:
            raise RuntimeError(f"Body '{self.args.base_body_name}' not found.")
        base_jid = int(self.m.body_jntadr[base_bid])

        l20_jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "l20_Joint")
        r20_jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "r20_Joint")
        lf0_jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "lf0_Joint")
        rf0_jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "rf0_Joint")
        if min(l20_jid, r20_jid, lf0_jid, rf0_jid) < 0:
            raise RuntimeError("Required parallel-leg joints not found.")

        return Handles(
            lf0=mj_actuator_id(self.m, "act_lf0"),
            lw=mj_actuator_id(self.m, "act_lw"),
            rf0=mj_actuator_id(self.m, "act_rf0"),
            rw=mj_actuator_id(self.m, "act_rw"),
            lf20=mj_actuator_id(self.m, "act_lf20"),
            rf20=mj_actuator_id(self.m, "act_rf20"),
            left_gas_spring=mj_actuator_id(self.m, LEFT_GAS_SPRING_ACTUATOR_NAME),
            right_gas_spring=mj_actuator_id(self.m, RIGHT_GAS_SPRING_ACTUATOR_NAME),
            base_bid=int(base_bid),
            base_qpos_adr=int(self.m.jnt_qposadr[base_jid]),
            l20_qpos_adr=int(self.m.jnt_qposadr[l20_jid]),
            r20_qpos_adr=int(self.m.jnt_qposadr[r20_jid]),
            l20_qvel_adr=int(self.m.jnt_dofadr[l20_jid]),
            r20_qvel_adr=int(self.m.jnt_dofadr[r20_jid]),
            lf0_qpos_adr=int(self.m.jnt_qposadr[lf0_jid]),
            rf0_qpos_adr=int(self.m.jnt_qposadr[rf0_jid]),
            lf0_qvel_adr=int(self.m.jnt_dofadr[lf0_jid]),
            rf0_qvel_adr=int(self.m.jnt_dofadr[rf0_jid]),
        )

    def init_sensor_cache(self) -> None:
        self.cache_sensor("base_ang_vel")
        for dof_name in DOF_NAMES:
            self.cache_sensor(f"{dof_name}_p")
            self.cache_sensor(f"{dof_name}_v")

    def cache_sensor(self, name: str) -> None:
        sensor_id = mj_sensor_id(self.m, name)
        self.sensor_cache[name] = (int(self.m.sensor_adr[sensor_id]), int(self.m.sensor_dim[sensor_id]))

    def get_sensor(self, name: str) -> np.ndarray:
        adr, dim = self.sensor_cache[name]
        return np.asarray(self.d.sensordata[adr : adr + dim], dtype=np.float32).copy()

    def get_base_quat_wxyz(self) -> np.ndarray:
        return np.asarray(self.d.xquat[self.handles.base_bid], dtype=np.float32).copy()

    def read_dof_pos(self) -> np.ndarray:
        q = np.zeros((6,), dtype=np.float32)
        for i, dof_name in enumerate(DOF_NAMES):
            q[i] = float(self.get_sensor(f"{dof_name}_p")[0])
        return q

    def estimate_vel(self, q: np.ndarray) -> np.ndarray:
        if self.vel_last_pos is None or self.vel_last_time is None:
            qd = np.zeros_like(q)
        else:
            dt = float(self.d.time) - self.vel_last_time
            if dt > 1e-12:
                diff = np.remainder(q - self.vel_last_pos + math.pi, 2.0 * math.pi) - math.pi
                qd = diff / dt
            else:
                qd = np.zeros_like(q)
        self.vel_last_pos = q.copy()
        self.vel_last_time = float(self.d.time)
        return qd.astype(np.float32)

    def read_virtual_leg_state(self) -> tuple[float, float, float, float]:
        lf0_angle = float(self.d.qpos[self.handles.lf0_qpos_adr])
        rf0_angle = -float(self.d.qpos[self.handles.rf0_qpos_adr])
        lf20_angle = float(self.d.qpos[self.handles.l20_qpos_adr])
        rf20_angle = -float(self.d.qpos[self.handles.r20_qpos_adr])

        lf0_vel = float(self.d.qvel[self.handles.lf0_qvel_adr])
        rf0_vel = -float(self.d.qvel[self.handles.rf0_qvel_adr])
        lf20_vel = float(self.d.qvel[self.handles.l20_qvel_adr])
        rf20_vel = -float(self.d.qvel[self.handles.r20_qvel_adr])

        dlf1_dphi1, dlf1_dphi4 = solve_phi_piandao(solve_phi3_left, DEFAULT_OFFSET + lf20_angle, lf0_angle, self.args.l1, self.args.l2, MAP_EPS)
        drf1_dphi1, drf1_dphi4 = solve_phi_piandao(solve_phi3_right, DEFAULT_OFFSET + rf20_angle, rf0_angle, self.args.l1, self.args.l2, MAP_EPS)
        lf1_pos = solve_phi3_left(DEFAULT_OFFSET + lf20_angle, lf0_angle, self.args.l1, self.args.l2)
        rf1_pos = solve_phi3_right(DEFAULT_OFFSET + rf20_angle, rf0_angle, self.args.l1, self.args.l2)
        lf1_vel = dlf1_dphi1 * lf20_vel + dlf1_dphi4 * lf0_vel
        rf1_vel = drf1_dphi1 * rf20_vel + drf1_dphi4 * rf0_vel
        return float(lf1_pos), float(rf1_pos), float(lf1_vel), float(rf1_vel)

    def build_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q_wxyz = self.get_base_quat_wxyz()
        q_xyzw = wxyz_to_xyzw(q_wxyz)
        base_ang = self.get_sensor("base_ang_vel").reshape(-1)
        q = self.read_dof_pos()
        qd = self.estimate_vel(q)

        lf1_pos, rf1_pos, lf1_vel, rf1_vel = self.read_virtual_leg_state()
        q[LF1_IDX] = lf1_pos
        q[RF1_IDX] = rf1_pos
        qd[LF1_IDX] = lf1_vel
        qd[RF1_IDX] = rf1_vel

        projected_g = quat_rotate_inverse_xyzw(q_xyzw, self.g_world)
        return base_ang, projected_g, q, qd

    def build_obs(self, base_ang: np.ndarray, projected_g: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if self.args.print_base_ang_vel:
            self.debug_counter += 1
            if self.debug_counter % max(1, int(self.args.print_interval)) == 0:
                print(f"[BASE_ANG_VEL] {np.array2string(base_ang, precision=4)}")
        cmd = np.array(
            [self.command_state.cmd_lin_vel, self.command_state.cmd_yaw_vel, self.command_state.cmd_height],
            dtype=np.float32,
        )
        cmd[2] = float(np.clip(cmd[2], HEIGHT_MIN, HEIGHT_MAX))
        q_obs = q[OBS_DOF_POS_IDXS]
        obs = np.concatenate(
            [
                base_ang * OBS_SCALE_ANG_VEL,
                projected_g,
                cmd * self.active_command_scale,
                (q_obs - self.active_default_obs_dof_pos) * OBS_SCALE_DOF_POS,
                qd * OBS_SCALE_DOF_VEL,
                self.last_actions,
            ],
            axis=0,
        ).astype(np.float32)
        return np.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS).astype(np.float32)

    def update_history(self, obs: np.ndarray) -> None:
        if self.history is None:
            self.history = np.repeat(obs.reshape(1, -1), HISTORY_LEN, axis=0)
        else:
            self.history[:-1, :] = self.history[1:, :]
            self.history[-1, :] = obs

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        feed = {self.obs_name: obs.reshape(1, -1).astype(np.float32)}
        if self.hist_name is not None and self.history is not None:
            feed[self.hist_name] = self.history.reshape(1, -1).astype(np.float32)
        action_raw = np.asarray(self.sess.run(None, feed)[0], dtype=np.float32).reshape(-1)
        actions = np.clip(fit_vector(action_raw, 6), -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        self.last_actions[:] = actions
        return actions

    def compute_control(self, actions: np.ndarray, q: np.ndarray, qd: np.ndarray) -> dict[str, float | np.ndarray]:
        pos_ref = actions * POS_ACTION_SCALE
        pos_ref[2] = 0.0
        pos_ref[5] = 0.0

        vel_ref = actions * VEL_ACTION_SCALE
        vel_ref[0] = 0.0
        vel_ref[1] = 0.0
        vel_ref[3] = 0.0
        vel_ref[4] = 0.0

        tau_virtual = self.active_p_gains * (pos_ref + self.active_default_dof_pos - q) + self.active_d_gains * (vel_ref - qd)
        # tau_virtual = np.clip(tau_virtual * float(self.args.torque_scale), -TAU_MAX, TAU_MAX).astype(np.float32)

        lf0_angle = float(self.d.qpos[self.handles.lf0_qpos_adr])
        rf0_angle = -float(self.d.qpos[self.handles.rf0_qpos_adr])
        lf20_angle = float(self.d.qpos[self.handles.l20_qpos_adr])
        rf20_angle = -float(self.d.qpos[self.handles.r20_qpos_adr])

        tau_lf0, tau_lf1 = float(tau_virtual[LF0_IDX]), float(tau_virtual[LF1_IDX])
        tau_rf0, tau_rf1 = float(tau_virtual[RF0_IDX]), float(tau_virtual[RF1_IDX])

        if self.args.torque_map == "numeric":
            pi_l_phi1, pi_l_phi4 = solve_phi_piandao(
                solve_phi3_left,
                DEFAULT_OFFSET + lf20_angle,
                lf0_angle,
                self.args.l1,
                self.args.l2,
                MAP_EPS,
            )
            pi_r_phi1, pi_r_phi4 = solve_phi_piandao(
                solve_phi3_right,
                DEFAULT_OFFSET + rf20_angle,
                rf0_angle,
                self.args.l1,
                self.args.l2,
                MAP_EPS,
            )
            tau_lf0_act = tau_lf0 + tau_lf1 * pi_l_phi4
            tau_lf20_act = tau_lf1 * pi_l_phi1
            tau_rf0_act = tau_rf0 - tau_rf1 * pi_r_phi4
            tau_rf20_act = -tau_rf1 * pi_r_phi1
            j_l = np.array([[1.0, 0.0], [pi_l_phi4, pi_l_phi1]], dtype=np.float32)
            j_r = np.array([[1.0, 0.0], [-pi_r_phi4, -pi_r_phi1]], dtype=np.float32)
        else:
            pi_l_phi1, pi_l_phi4 = solve_jacobian_analytic(DEFAULT_OFFSET + lf20_angle, lf0_angle, self.args.l1, self.args.l2)
            pi_r_phi1, pi_r_phi4 = solve_jacobian_analytic(DEFAULT_OFFSET + rf20_angle, rf0_angle, self.args.l1, self.args.l2)
            tau_lf0_act = tau_lf0 + tau_lf1 * pi_l_phi4
            tau_lf20_act = tau_lf1 * pi_l_phi1
            tau_rf0_act = tau_rf0 + tau_rf1 * pi_r_phi4
            tau_rf20_act = tau_rf1 * pi_r_phi1
            j_l = np.array([[1.0, 0.0], [pi_l_phi4, pi_l_phi1]], dtype=np.float32)
            j_r = np.array([[1.0, 0.0], [pi_r_phi4, pi_r_phi1]], dtype=np.float32)

        if self.args.print_jacobian:
            self.jacobian_debug_counter += 1
            if self.jacobian_debug_counter % max(1, int(self.args.print_interval)) == 0:
                print(
                    f"[JACOBIAN:{self.args.torque_map}] "
                    f"L={np.array2string(j_l, precision=5, suppress_small=True)} "
                    f"R={np.array2string(j_r, precision=5, suppress_small=True)}"
                )

        matrix_l, matrix_l_inv, left_l0 = force_map(DEFAULT_OFFSET + lf20_angle, lf0_angle, self.args.l1, self.args.l2)
        ftp_l = matrix_l_inv @ np.array([[tau_lf20_act], [tau_lf0_act]], dtype=np.float32)
        ftp_l[0, 0] -= GAS_SPRING_FORCE * left_l0
        if self.is_jump_f_scale_active():
            ftp_l[0, 0] *= float(self.args.jump_f_scale)
        tau_lf20_act, tau_lf0_act = (matrix_l @ ftp_l).reshape(-1).tolist()

        matrix_r, matrix_r_inv, right_l0 = force_map(DEFAULT_OFFSET + rf20_angle, rf0_angle, self.args.l1, self.args.l2)
        ftp_r = matrix_r_inv @ np.array([[tau_rf20_act], [tau_rf0_act]], dtype=np.float32)
        ftp_r[0, 0] += GAS_SPRING_FORCE * right_l0
        if self.is_jump_f_scale_active():
            ftp_r[0, 0] *= float(self.args.jump_f_scale)
        tau_rf20_act, tau_rf0_act = (matrix_r @ ftp_r).reshape(-1).tolist()

        tau_cmd = np.clip(
            np.array([tau_lf0_act, tau_virtual[LW_IDX], tau_rf0_act, tau_virtual[RW_IDX], tau_lf20_act, tau_rf20_act], dtype=np.float32),
            -TAU_MAX,
            TAU_MAX,
        ).astype(np.float32)

        tau_cmd[1] = np.clip(tau_cmd[1], -5.0 , 5.0).astype(np.float32)
        tau_cmd[3] = np.clip(tau_cmd[3], -5.0 , 5.0).astype(np.float32)
        return {
            "tau_lf0": float(tau_cmd[0]),
            "tau_lw": float(tau_cmd[1]),
            "tau_rf0": float(tau_cmd[2]),
            "tau_rw": float(tau_cmd[3]),
            "tau_lf20": float(tau_cmd[4]),
            "tau_rf20": float(tau_cmd[5]),
            "tau_cmd": tau_cmd,
        }

    def is_jump_f_scale_active(self) -> bool:
        return (
            self.jump_f_scale_start_time is not None
            and self.jump_f_scale_end_time is not None
            and self.jump_f_scale_start_time <= float(self.d.time) < self.jump_f_scale_end_time
        )

    def apply_ctrl(self, ctrl: dict[str, float | np.ndarray]) -> None:
        self.d.ctrl[:] = 0.0
        if not self.command_state.torque_enabled:
            return
        pairs = (
            (self.handles.lf0, "tau_lf0"),
            (self.handles.lw, "tau_lw"),
            (self.handles.rf0, "tau_rf0"),
            (self.handles.rw, "tau_rw"),
            (self.handles.lf20, "tau_lf20"),
            (self.handles.rf20, "tau_rf20"),
        )
        for actuator_id, key in pairs:
            value = float(ctrl[key])
            self.d.ctrl[actuator_id] = float(np.clip(clip_actuator_ctrl(self.m, actuator_id, value), -TAU_MAX, TAU_MAX))
        self.d.ctrl[self.handles.left_gas_spring] = clip_actuator_ctrl(self.m, self.handles.left_gas_spring, LEFT_GAS_SPRING_CTRL)
        self.d.ctrl[self.handles.right_gas_spring] = clip_actuator_ctrl(self.m, self.handles.right_gas_spring, RIGHT_GAS_SPRING_CTRL)

    def configure_camera(self, viewer: mujoco.viewer.Handle) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = self.d.xpos[self.handles.base_bid]
        viewer.cam.lookat[2] += 0.15
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -20.0

    def update_camera(self, viewer: mujoco.viewer.Handle) -> None:
        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            viewer.cam.lookat[:] = self.d.xpos[self.handles.base_bid]
            viewer.cam.lookat[2] += 0.15

    def run(self) -> None:
        listener = keyboard.Listener(on_press=self.command_state.on_press, on_release=self.command_state.on_release)
        listener.start()
        try:
            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                self.configure_camera(viewer)
                while viewer.is_running() and self.command_state.running:
                    self.handle_policy_requests()
                    self.handle_temporary_policy_return()
                    self.command_state.update_ramp(self.eff_policy_dt, self.args.cmd_ramp_time)
                    base_ang, projected_g, q, qd = self.build_state()
                    obs = self.build_obs(base_ang, projected_g, q, qd)
                    self.update_history(obs)
                    actions = self.run_policy(obs)

                    q_step, qd_step = q, qd
                    for step_idx in range(self.steps_per_policy):
                        time.sleep(self.sim_dt*2)
                        ctrl = self.compute_control(actions, q_step, qd_step)
                        self.apply_ctrl(ctrl)
                        mujoco.mj_step(self.m, self.d)
                        if step_idx < self.steps_per_policy - 1:
                            _, _, q_step, qd_step = self.build_state()

                    self.update_camera(viewer)
                    viewer.sync()
        finally:
            listener.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML_PATH)
    parser.add_argument("--onnx", type=str, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy_dt", type=float, default=0.01)
    parser.add_argument("--sim_dt", type=float, default=0.005)
    parser.add_argument("--cmd_ramp_time", type=float, default=CMD_RAMP_TIME)
    parser.add_argument("--jump_onnx", type=str, default=JUMP_ONNX_PATH)
    parser.add_argument("--jump_policy_duration", type=float, default=JUMP_POLICY_DURATION_S)
    parser.add_argument("--torque_scale", type=float, default=1.0)
    parser.add_argument("--torque_map", type=str, default="analytic", choices=["analytic", "numeric"])
    parser.add_argument("--base_body_name", type=str, default="base_Link_del")
    parser.add_argument("--l1", type=float, default=0.175)
    parser.add_argument("--l2", type=float, default=0.208)
    parser.add_argument("--print_base_ang_vel", action="store_true")
    parser.add_argument("--print_jacobian", action="store_true")
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--jump_f_scale", type=float, default=FFF)
    parser.add_argument("--jump_f_scale_start", type=float, default=XXX)
    parser.add_argument("--jump_f_scale_end", type=float, default=YYY)
    parser.add_argument("--tag", nargs="?", const=True, default=Jump_after_height_big, type=parse_bool_flag)
    parser.add_argument("--jump_after_height", type=float, default=Jump_after_height)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    RUNTIME_PRESETS[JUMP_POLICY_NAME]["onnx_path"] = args.jump_onnx
    command_state = CommandState(jump_policy_duration_s=max(0.0, float(args.jump_policy_duration)))
    runtime = BinglianRuntime(args, command_state)
    runtime.run()


if __name__ == "__main__":
    main()
