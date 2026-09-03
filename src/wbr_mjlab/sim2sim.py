"""Native MuJoCo + ONNX deployment, without an RL environment or Torch inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from .robot import (
  ACTIVE_JOINT_NAMES,
  HOME_ACTIVE_JOINT_POS,
  LEG_IDS,
  MAX_FORWARD_COMMAND,
  MAX_YAW_COMMAND,
  MOTOR_ACTUATOR_NAMES,
  POLICY_DT,
  TORQUE_LIMITS,
  WBR_XML_PATH,
  WHEEL_GEOM_NAMES,
  load_wbr_spec,
)
from .task import make_env_cfg
from .terrain import attach_xml_terrain, configured_terrain_xml, resolve_terrain_xml

METADATA_KEY = "wbr_contract"


def policy_contract(mode: str) -> dict:
  if mode not in ("plane", "jump"):
    raise ValueError(f"Unknown WBR mode: {mode}")
  return {
    "version": 1,
    "mode": mode,
    "joint_names": list(ACTIVE_JOINT_NAMES),
    "default_q": list(HOME_ACTIVE_JOINT_POS),
    "torque_limits": list(TORQUE_LIMITS),
    "physics_dt": 0.001,
    "policy_dt": POLICY_DT,
    "obs_dim": 25,
    "history_frames": 5,
    "history_order": "oldest_first_unclipped",
    "velocity": "wrapped_qpos_difference_per_physics_step",
    "leg_kp_kd": [20.0, 1.0] if mode == "plane" else [6.0, 0.5],
    "wheel_kd": 0.2,
    "action_scales": [0.5, 10.0],
    "command_scales": [2.0 if mode == "plane" else 3.0, 0.25, 5.0],
    "mjcf_sha256": hashlib.sha256(WBR_XML_PATH.read_bytes()).hexdigest(),
  }


class OnnxPolicy:
  def __init__(self, path: Path, mode: str):
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise RuntimeError(
        "Install ONNX Runtime: python -m pip install 'onnxruntime>=1.19,<1.24'"
      ) from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = {value.name: value for value in self.session.get_inputs()}
    outputs = self.session.get_outputs()
    if set(inputs) != {"obs", "obs_history"} or len(outputs) != 1:
      raise ValueError("Expected WBR inputs obs, obs_history and one actions output")
    for value, width in ((inputs["obs"], 25), (inputs["obs_history"], 125), (outputs[0], 6)):
      if (
        value.type != "tensor(float)"
        or len(value.shape) != 2
        or value.shape[1] != width
        or (isinstance(value.shape[0], int) and value.shape[0] != 1)
      ):
        raise ValueError(f"Invalid WBR tensor: {value.name} {value.type} {value.shape}")
    if outputs[0].name != "actions":
      raise ValueError("Expected ONNX output named actions")
    metadata = self.session.get_modelmeta().custom_metadata_map
    if METADATA_KEY in metadata:
      if json.loads(metadata[METADATA_KEY]) != policy_contract(mode):
        raise ValueError(
          "ONNX mode/model/controller contract mismatch; re-export with the correct --task"
        )
    else:
      warnings.warn(
        "ONNX has no WBR metadata; --mode and training model must be checked manually", stacklevel=2
      )

  def __call__(self, obs: np.ndarray, history: np.ndarray) -> np.ndarray:
    if not np.isfinite(obs).all() or not np.isfinite(history).all():
      raise FloatingPointError("Non-finite policy input")
    action = self.session.run(["actions"], {"obs": obs, "obs_history": history})[0]
    if action.shape != (1, 6) or not np.isfinite(action).all():
      raise FloatingPointError("Invalid ONNX action")
    return np.clip(action[0], -100.0, 100.0)


def observation_frame(ang_vel, quat, command, q, qd, action, mode: str) -> np.ndarray:
  """Exactly the training policy_vector, before current-frame clipping."""
  quat = np.asarray(quat, dtype=np.float32)
  gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
  gravity += 2 * np.cross(quat[1:], np.cross(quat[1:], gravity) - quat[0] * gravity)
  scales = np.array([2.0 if mode == "plane" else 3.0, 0.25, 5.0], dtype=np.float32)
  return np.concatenate(
    (
      np.asarray(ang_vel, dtype=np.float32) * 0.25,
      gravity,
      np.asarray(command, dtype=np.float32) * scales,
      (q - np.asarray(HOME_ACTIVE_JOINT_POS, dtype=np.float32))[list(LEG_IDS)],
      qd * 0.05,
      action,
    )
  ).astype(np.float32)


def motor_torque(action, q, qd, mode: str) -> np.ndarray:
  kp, kd = (20.0, 1.0) if mode == "plane" else (6.0, 0.5)
  target = np.asarray(HOME_ACTIVE_JOINT_POS, dtype=np.float32) + 0.5 * action
  torque = kp * (target - q) - kd * qd
  torque[[2, 5]] = 0.2 * (10.0 * action[[2, 5]] - qd[[2, 5]])
  return np.clip(torque, -np.asarray(TORQUE_LIMITS), TORQUE_LIMITS)


def grounded_wheel_count(
  contacts, terrain_geom_ids: int | frozenset[int], wheel_geom_ids: frozenset[int]
) -> int:
  """Count wheel collision geoms currently touching the terrain."""
  terrain_ids = (
    frozenset((terrain_geom_ids,)) if isinstance(terrain_geom_ids, int) else terrain_geom_ids
  )
  grounded: set[int] = set()
  for contact in contacts:
    geom1, geom2 = int(contact.geom1), int(contact.geom2)
    if geom1 in terrain_ids and geom2 in wheel_geom_ids:
      grounded.add(geom2)
    elif geom2 in terrain_ids and geom1 in wheel_geom_ids:
      grounded.add(geom1)
  return len(grounded)


def wheels_grounded(
  contacts, terrain_geom_ids: int | frozenset[int], wheel_geom_ids: frozenset[int]
) -> bool:
  """Return true only when both wheel collision geoms touch the terrain."""
  return grounded_wheel_count(contacts, terrain_geom_ids, wheel_geom_ids) == len(wheel_geom_ids)


def native_model(mode: str, terrain_xml: str | Path | None = None) -> mujoco.MjModel:
  """Use the same mesh, closed chains, gas springs, floor and options as training."""
  terrain_xml_path = configured_terrain_xml(terrain_xml)
  cfg = make_env_cfg(mode, play=True, terrain_xml=terrain_xml_path)
  spec = load_wbr_spec()
  if terrain_xml_path is None:
    spec.add_texture(
      name="ground_grid",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      rgb1=[0.32, 0.40, 0.48],
      rgb2=[0.56, 0.63, 0.69],
      width=256,
      height=256,
    )
    material = spec.add_material(
      name="ground_grid",
      texuniform=True,
      texrepeat=[2, 2],
      reflectance=0.1,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ground_grid"
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=[0.35, 0.50, 0.65],
    rgb2=[0.85, 0.90, 0.95],
    width=256,
    height=1536,
  )
  if terrain_xml_path is None:
    spec.worldbody.add_geom(
      name="terrain",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[0, 0, 0.05],
      contype=1,
      conaffinity=2,
      friction=[0.8, 0.005, 0.0001],
      material="ground_grid",
    )
  else:
    attach_xml_terrain(spec, terrain_xml_path)
  spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
  model = spec.compile()
  cfg.sim.mujoco.apply(model)
  return model


class NativeRunner:
  def __init__(
    self,
    policy,
    mode: str,
    *,
    policies: dict | None = None,
    terrain_xml: str | Path | None = None,
  ):
    self.mode = mode
    self.policy = policy
    self.policies = dict(policies or {})
    self.policies[mode] = policy
    self.model = native_model(mode, terrain_xml)
    self.data = mujoco.MjData(self.model)
    self.qadr = np.array([self.model.joint(name).qposadr[0] for name in ACTIVE_JOINT_NAMES])
    self.motor_ids = np.array([self.model.actuator(name).id for name in MOTOR_ACTUATOR_NAMES])
    self.terrain_geom_ids = frozenset(
      geom_id
      for geom_id in range(self.model.ngeom)
      if self.model.geom(geom_id).name == "terrain"
      or self.model.geom(geom_id).name.startswith("terrain/")
    )
    if not self.terrain_geom_ids:
      raise ValueError("Simulation contains no terrain geoms")
    # Retain the singular attribute for callers using the default plane.
    self.terrain_geom_id = next(iter(self.terrain_geom_ids))
    self.wheel_geom_ids = frozenset(self.model.geom(name).id for name in WHEEL_GEOM_NAMES)
    root_joint = self.model.body("base_link").jntadr[0]
    self.root_qadr = int(self.model.jnt_qposadr[root_joint])
    self.root_vadr = int(self.model.jnt_dofadr[root_joint])
    self.decimation = round(POLICY_DT / self.model.opt.timestep)
    self.reset()

  def wheels_grounded(self) -> bool:
    return wheels_grounded(self.data.contact, self.terrain_geom_ids, self.wheel_geom_ids)

  def grounded_wheel_count(self) -> int:
    return grounded_wheel_count(self.data.contact, self.terrain_geom_ids, self.wheel_geom_ids)

  def switch_policy(self, mode: str) -> bool:
    if mode not in self.policies:
      raise ValueError(f"No {mode} policy loaded; provide --{mode}-onnx")
    if mode == self.mode:
      return False
    self.policy = self.policies[mode]
    self.mode = mode
    # Both tasks share physical state and action units, but command scaling and
    # PD gains differ. Backfill fresh history on the next inference; retain the
    # actual previous action and finite-difference velocity through the handover.
    self.history = None
    return True

  def reset(self) -> None:
    mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
    mujoco.mj_forward(self.model, self.data)
    self.q = self.data.qpos[self.qadr].astype(np.float32)
    self.qd = np.zeros(6, dtype=np.float32)
    self.action = np.zeros(6, dtype=np.float32)
    self.torque = np.zeros(6, dtype=np.float32)
    self.history = None
    self.last_obs = np.zeros((1, 25), dtype=np.float32)

  def observe(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = observation_frame(
      self.data.qvel[self.root_vadr + 3 : self.root_vadr + 6],
      self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7],
      command,
      self.q,
      self.qd,
      self.action,
      self.mode,
    )
    if self.history is None:
      self.history = np.tile(frame, (5, 1))
    else:
      self.history[:-1] = self.history[1:]
      self.history[-1] = frame
    self.last_obs = np.clip(frame[None], -100.0, 100.0)
    return self.last_obs, self.history.reshape(1, 125).copy()

  def step(self, command: np.ndarray, *, enabled: bool = True) -> None:
    if enabled:
      # Observation history is part of the policy transaction. A failed external
      # backend (notably USB HIL) must neither consume a history frame nor leave the
      # previously applied action armed for a later retry.
      previous_history = None if self.history is None else self.history.copy()
      previous_obs = self.last_obs.copy()
      try:
        next_action = np.asarray(self.policy(*self.observe(command)), dtype=np.float32)
      except BaseException:
        self.history = previous_history
        self.last_obs = previous_obs
        self.action.fill(0)
        self.torque.fill(0)
        self.data.ctrl[:] = 0
        raise
      self.action = next_action
      if self.action.shape != (6,) or not np.isfinite(self.action).all():
        self.history = previous_history
        self.last_obs = previous_obs
        self.action = np.zeros(6, dtype=np.float32)
        self.torque.fill(0)
        self.data.ctrl[:] = 0
        raise FloatingPointError("Invalid policy action")
      self.action = np.clip(self.action, -100.0, 100.0)
    else:
      self.action.fill(0)
      self.history = None
    for _ in range(self.decimation):
      self.torque = (
        motor_torque(self.action, self.q, self.qd, self.mode) if enabled else np.zeros(6)
      )
      self.data.ctrl[:] = 0  # Gas spring bias stays active; motor disable is not a freeze.
      self.data.ctrl[self.motor_ids] = self.torque
      mujoco.mj_step(self.model, self.data)
      if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
        raise FloatingPointError("Non-finite MuJoCo state")
      for flag in (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
      ):
        if self.data.warning[flag].number:
          raise FloatingPointError(f"MuJoCo numerical warning: {flag.name}")
      q = self.data.qpos[self.qadr].astype(np.float32)
      # Sample once per newly integrated state. Reuse this velocity for the next
      # observation AND next PD substep, as WbrState's qpos-version cache does.
      self.qd = ((q - self.q + np.pi) % (2 * np.pi) - np.pi) / self.model.opt.timestep
      self.q = q
    mujoco.mj_forward(self.model, self.data)


class KeyboardControl:
  """GLFW press/release state, following wbr_control's input semantics."""

  def __init__(
    self, mode: str, velocity: float = 0.8, yaw_rate: float = 1.5, *, available_modes=None
  ):
    self.mode = mode
    self.available_modes = frozenset(available_modes or (mode,))
    self.velocity, self.yaw_rate = velocity, yaw_rate
    self.reset()

  @property
  def heights(self) -> tuple[float, float, float]:
    return (0.10, 0.15, 0.20) if self.mode == "plane" else (0.12, 0.135, 0.15)

  def reset(self) -> None:
    self.held: set[int] = set()
    self.height = self.heights[1]
    self.enabled = False
    self.paused = True
    self.reset_requested = False
    self.quit_requested = False
    self.pending_mode: str | None = None
    self.pending_jump_once = False
    self.jump_once_phase: str | None = None
    self.jump_once_ground_seen = False
    self.jump_once_air_steps = 0
    self.jump_once_land_steps = 0
    self.notice = "Enter: enable and run"

  def _cancel_jump_once(self) -> None:
    self.pending_jump_once = False
    self.jump_once_phase = None
    self.jump_once_ground_seen = False
    self.jump_once_air_steps = 0
    self.jump_once_land_steps = 0

  def request_policy(self, mode: str, *, jump_once: bool = False) -> None:
    if mode not in self.available_modes:
      self.notice = f"No {mode} policy loaded; add --{mode}-onnx PATH"
      print(self.notice)
      return
    if not jump_once:
      self._cancel_jump_once()
    self.pending_mode = mode
    self.pending_jump_once = jump_once

  def apply_policy_request(self, runner: NativeRunner) -> None:
    if self.pending_mode is None:
      return
    mode, self.pending_mode = self.pending_mode, None
    jump_once, self.pending_jump_once = self.pending_jump_once, False
    if runner.switch_policy(mode):
      self.mode = mode
      self.height = self.heights[1]
    if jump_once and mode == "jump":
      self.jump_once_phase = "takeoff"
      self.jump_once_ground_seen = runner.wheels_grounded()
      self.jump_once_air_steps = 0
      self.jump_once_land_steps = 0
      self.notice = "One-shot jump: waiting for takeoff"
    else:
      self._cancel_jump_once()
      self.notice = f"Active policy: {self.mode} (Space one-shot jump, 1 plane, 2 jump)"
    print(self.notice)

  def update_jump_once(self, runner: NativeRunner) -> None:
    if self.jump_once_phase is None or self.mode != "jump" or not self.enabled:
      return
    contact_count = runner.grounded_wheel_count()
    grounded = contact_count == 2
    airborne = contact_count == 0
    if self.jump_once_phase == "takeoff":
      if grounded:
        self.jump_once_ground_seen = True
        self.jump_once_air_steps = 0
      elif airborne and self.jump_once_ground_seen:
        self.jump_once_air_steps += 1
        if self.jump_once_air_steps >= 2:
          self.jump_once_phase = "landing"
          self.jump_once_land_steps = 0
          self.notice = "One-shot jump: airborne, waiting for landing"
      else:
        self.jump_once_air_steps = 0
      return
    if self.jump_once_phase == "landing":
      self.jump_once_land_steps = self.jump_once_land_steps + 1 if grounded else 0
      if self.jump_once_land_steps >= 3:
        self.jump_once_phase = "returning"
        self.pending_mode = "plane"
        self.notice = "One-shot jump: landed, returning to plane"

  def key(self, key: int, action: int) -> None:
    # GLFW: release=0, press=1, repeat=2. Only physical press edges toggle modes.
    if action == 0:
      self.held.discard(key)
      return
    if action != 1 or key in self.held:
      return
    self.held.add(key)
    if key == 257:  # Enter
      self.enabled = not self.enabled
      self.paused = False
      self.notice = (
        "Policy enabled" if self.enabled else "Motors disabled (gas springs remain active)"
      )
    elif key in (ord("Q"), ord("E"), ord("F")):
      self.height = self.heights[(ord("Q"), ord("E"), ord("F")).index(key)]
    elif key == ord("P"):
      self.paused = not self.paused
    elif key == 259:  # Backspace
      self.reset_requested = True
    elif key == ord("X"):
      self.held.clear()
    elif key == 256:  # Escape
      self.quit_requested = True
    elif key == 32:
      current = self.pending_mode or self.mode
      if self.jump_once_phase is not None or self.pending_jump_once:
        self.notice = "One-shot jump already active"
      elif current == "plane":
        self.request_policy("jump", jump_once=True)
      else:
        self.request_policy("plane")
    elif key in (ord("1"), ord("2")):
      self.request_policy("plane" if key == ord("1") else "jump")
    elif key == ord("G"):
      self.notice = "No stair policy loaded; G does not apply forces"
      print(self.notice)

  def lose_focus(self) -> None:
    self.held.clear()  # Do not leave a movement key latched when focus is lost.

  def command(self) -> np.ndarray:
    down = self.held.__contains__
    vx = self.velocity * (down(ord("W")) - down(ord("S")))
    yaw = self.yaw_rate * (down(ord("A")) - down(ord("D")))
    if down(340) or down(344):
      vx, yaw = 0.0, self.yaw_rate
    if not self.enabled:
      vx = yaw = 0.0
    return np.array([vx, yaw, self.height], dtype=np.float32)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--onnx", type=Path, help="ONNX for the initial --mode (single-policy syntax)"
  )
  parser.add_argument("--plane-onnx", type=Path, help="Plane policy for keyboard switching")
  parser.add_argument("--jump-onnx", type=Path, help="Jump policy for keyboard switching")
  parser.add_argument("--mode", default="plane", choices=("plane", "jump"), help="Initial policy")
  parser.add_argument(
    "--gpu",
    choices=("nvidia", "system"),
    default="nvidia",
    help="OpenGL GPU for the viewer (default: NVIDIA PRIME offload)",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=2000, help="Headless policy steps (100 Hz)")
  parser.add_argument("--vx", type=float, default=0.0, help="Headless forward command, m/s")
  parser.add_argument("--yaw", type=float, default=0.0, help="Headless yaw command, rad/s")
  parser.add_argument("--height", type=float, help="Headless root-height command, m")
  parser.add_argument("--velocity", type=float, default=0.8, help="Keyboard speed, m/s")
  parser.add_argument("--yaw-rate", type=float, default=1.5, help="Keyboard turn/spin speed, rad/s")
  parser.add_argument(
    "--terrain-xml", type=Path, help="Static MuJoCo XML/MJCF terrain (relative to project root)"
  )
  parser.add_argument("--output", type=Path, help="Headless trajectory .npz")
  args = parser.parse_args(argv)
  if args.terrain_xml is not None:
    try:
      args.terrain_xml = resolve_terrain_xml(args.terrain_xml)
    except (FileNotFoundError, ValueError) as exc:
      parser.error(str(exc))
  args.policy_paths = {
    mode: path
    for mode, path in (("plane", args.plane_onnx), ("jump", args.jump_onnx))
    if path is not None
  }
  if args.onnx is not None:
    if args.mode in args.policy_paths:
      parser.error(f"Use either --onnx or --{args.mode}-onnx for the initial policy, not both")
    args.policy_paths[args.mode] = args.onnx
  if args.mode not in args.policy_paths:
    parser.error(f"Provide --{args.mode}-onnx or --onnx for the initial --mode {args.mode}")
  heights = (0.10, 0.20) if args.mode == "plane" else (0.12, 0.15)
  args.height = args.height if args.height is not None else sum(heights) / 2
  if not (
    args.steps > 0
    and abs(args.vx) <= MAX_FORWARD_COMMAND
    and abs(args.yaw) <= MAX_YAW_COMMAND
    and heights[0] <= args.height <= heights[1]
    and 0 < args.velocity <= MAX_FORWARD_COMMAND
    and 0 < args.yaw_rate <= MAX_YAW_COMMAND
  ):
    parser.error("Commands must be finite and within configured limits; steps must be positive")
  return args


def main() -> None:
  args = parse_args()
  policies = {mode: OnnxPolicy(path, mode) for mode, path in args.policy_paths.items()}
  runner = NativeRunner(
    policies[args.mode], args.mode, policies=policies, terrain_xml=args.terrain_xml
  )
  print(f"Native MuJoCo / ONNX CPU | {args.mode} | physics 1000 Hz / policy 100 Hz")
  if not args.headless:
    from .sim2sim_viewer import run_viewer

    run_viewer(
      runner,
      KeyboardControl(
        args.mode,
        args.velocity,
        args.yaw_rate,
        available_modes=policies,
      ),
      gpu=args.gpu,
    )
    return
  command = np.array([args.vx, args.yaw, args.height], dtype=np.float32)
  trajectory = {
    name: [] for name in ("time", "qpos", "qvel", "command", "obs", "history", "action", "torque")
  }
  start = time.perf_counter()
  min_height = float("inf")
  for _ in range(args.steps):
    runner.step(command)
    min_height = min(min_height, runner.data.qpos[runner.root_qadr + 2])
    if args.output:
      for name, value in zip(
        trajectory,
        (
          runner.data.time,
          runner.data.qpos,
          runner.data.qvel,
          command,
          runner.last_obs[0],
          runner.history.reshape(125),
          runner.action,
          runner.torque,
        ),
        strict=True,
      ):
        trajectory[name].append(np.array(value, copy=True))
  elapsed = time.perf_counter() - start
  result = {
    "mode": args.mode,
    "onnx": str(args.policy_paths[args.mode]),
    "steps": args.steps,
    "sim_seconds": runner.data.time,
    "wall_seconds": elapsed,
    "realtime_factor": runner.data.time / elapsed,
    "min_root_height": min_height,
    "final_root_pos": runner.data.qpos[runner.root_qadr : runner.root_qadr + 3].tolist(),
    "finite": True,
  }
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **trajectory, metadata=json.dumps(result))
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
