"""Shared WBR task dynamics, observations, rewards and randomization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from .rewards import joint_position_limit_penalty as joint_position_limit_penalty
from .rewards import reward_value
from .rewards import weighted_clipped_reward as weighted_clipped_reward
from .robot import (
  ACTIVE_JOINT_NAMES,
  HOME_ACTIVE_JOINT_POS,
  LEG_IDS,
  LEG_LINK_1,
  LEG_LINK_2,
  MOTOR_ZERO_RAD,
  POLICY_DT,
  TORQUE_LIMITS,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

Mode = Literal["plane", "jump"]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vectors by inverse unit quaternions in MuJoCo wxyz order."""
  qw, qv = q[..., :1], q[..., 1:]
  return v + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=-1) - qw * v, dim=-1)


def virtual_leg_geometry(
  active_q: torch.Tensor, *, motor_zero: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """WBR five-bar length and angle using wbr_control signs and mechanical zeros."""
  zero = active_q.new_tensor(MOTOR_ZERO_RAD) if motor_zero is None else motor_zero
  q = active_q - zero
  phi1 = torch.stack((math.pi - q[:, 0], math.pi + q[:, 3]), dim=1)
  phi4 = torch.stack((-q[:, 1], q[:, 4]), dim=1)
  xdb = LEG_LINK_1 * (torch.cos(phi4) - torch.cos(phi1))
  ydb = LEG_LINK_1 * (torch.sin(phi4) - torch.sin(phi1))
  a0, b0 = 2.0 * LEG_LINK_2 * xdb, 2.0 * LEG_LINK_2 * ydb
  c0 = xdb.square() + ydb.square()
  disc = (a0.square() + b0.square() - c0.square()).clamp_min(0.0)
  u2 = 2.0 * torch.atan2(b0 + torch.sqrt(disc), a0 + c0)
  cx = LEG_LINK_1 * torch.cos(phi1) + LEG_LINK_2 * torch.cos(u2)
  cy = LEG_LINK_1 * torch.sin(phi1) + LEG_LINK_2 * torch.sin(u2)
  length = torch.sqrt(cx.square() + cy.square())
  angle = torch.atan2(cy, cx) - 0.5 * math.pi
  angle = torch.atan2(torch.sin(angle), torch.cos(angle))
  return length, angle


def filtered_flight_state(
  contact: torch.Tensor, previous_contact: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  filtered = contact | previous_contact
  return filtered, torch.all(~filtered, dim=1)


def update_fall_counter(count: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
  return torch.where(bad, count + 1, torch.zeros_like(count))


def hybrid_constants(
  like: torch.Tensor, mode: Mode
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Allocate controller constants once, outside the physics substep loop."""
  kp, kd = (20.0, 1.0) if mode == "plane" else (6.0, 0.5)
  return (
    like.new_tensor((kp, kp, 0.0, kp, kp, 0.0)),
    like.new_tensor((kd, kd, 0.2, kd, kd, 0.2)),
    like.new_tensor(TORQUE_LIMITS),
    torch.tensor((2, 5), dtype=torch.long, device=like.device),
  )


def hybrid_torque(
  action: torch.Tensor,
  q: torch.Tensor,
  qd: torch.Tensor,
  default_q: torch.Tensor,
  kp_scale: torch.Tensor,
  kd_scale: torch.Tensor,
  torque_scale: torch.Tensor,
  mode: Mode,
  *,
  constants: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
  """Original four position-PD legs plus two velocity-PD wheels."""
  base_kp, base_kd, limits, wheel_ids = (
    hybrid_constants(action, mode) if constants is None else constants
  )
  target_q = default_q + 0.5 * action
  target_qd = 10.0 * action
  torque = base_kp * kp_scale * (target_q - q) - base_kd * kd_scale * qd
  torque[:, wheel_ids] = (
    base_kd[wheel_ids] * kd_scale[:, wheel_ids] * (target_qd[:, wheel_ids] - qd[:, wheel_ids])
  )
  torque *= torque_scale
  return torch.maximum(torch.minimum(torque, limits), -limits)


def policy_vector(
  ang_vel: torch.Tensor,
  gravity: torch.Tensor,
  command: torch.Tensor,
  active_q: torch.Tensor,
  default_q: torch.Tensor,
  active_qd: torch.Tensor,
  action: torch.Tensor,
  mode: Mode,
  *,
  command_scale: torch.Tensor | None = None,
) -> torch.Tensor:
  if command_scale is None:
    command_scale = command.new_tensor((2.0 if mode == "plane" else 3.0, 0.25, 5.0))
  return torch.cat(
    (
      ang_vel * 0.25,
      gravity,
      command * command_scale,
      (active_q[:, LEG_IDS] - default_q[:, LEG_IDS]),
      active_qd * 0.05,
      action,
    ),
    dim=-1,
  )


def policy_noise_scales(like: torch.Tensor) -> torch.Tensor:
  return like.new_tensor((0.05,) * 6 + (0.0,) * 3 + (0.02,) * 4 + (0.075,) * 6 + (0.0,) * 6)


def add_policy_noise(obs: torch.Tensor, *, scales: torch.Tensor | None = None) -> torch.Tensor:
  if scales is None:
    scales = policy_noise_scales(obs)
  return obs + (2.0 * torch.rand_like(obs) - 1.0) * scales


def critic_vector(
  base_lin_vel: torch.Tensor,
  clean_policy: torch.Tensor,
  previous_action: torch.Tensor,
  previous_previous_action: torch.Tensor,
  dof_acc: torch.Tensor,
  root_height: torch.Tensor,
  torque: torch.Tensor,
  base_mass: torch.Tensor,
  base_com: torch.Tensor,
  default_offset: torch.Tensor,
  friction: torch.Tensor,
  restitution: torch.Tensor,
  mode: Mode,
) -> torch.Tensor:
  heights = (root_height[:, None] - 0.5).clamp(-1.0, 1.0).repeat(1, 77) * 5.0
  return torch.cat(
    (
      base_lin_vel * (2.0 if mode == "plane" else 3.0),
      clean_policy,
      previous_action,
      previous_previous_action,
      dof_acc * 0.0025,
      heights,
      torque * 0.05,
      (base_mass - base_mass.mean())[:, None],
      base_com,
      default_offset,
      friction[:, None],
      restitution[:, None],
    ),
    dim=-1,
  )


@dataclass
class WbrState:
  env: ManagerBasedRlEnv
  mode: Mode

  def __post_init__(self) -> None:
    robot: Entity = self.env.scene["robot"]
    ids, names = robot.find_joints(ACTIVE_JOINT_NAMES, preserve_order=True)
    if tuple(names) != ACTIVE_JOINT_NAMES:
      raise ValueError(f"WBR motor order mismatch: {names}")
    self.robot = robot
    self.joint_ids = torch.tensor(ids, device=self.env.device)
    self.q_adr = robot.indexing.joint_q_adr[self.joint_ids]
    n, dev = self.env.num_envs, self.env.device
    self.raw_default_q = torch.tensor(HOME_ACTIVE_JOINT_POS, device=dev).repeat(n, 1)
    self.motor_zero = self.raw_default_q.new_tensor(MOTOR_ZERO_RAD)
    self.gravity = self.raw_default_q.new_tensor((0.0, 0.0, -1.0)).expand(n, -1)
    self.command_scale = self.raw_default_q.new_tensor(
      (2.0 if self.mode == "plane" else 3.0, 0.25, 5.0)
    )
    self.noise_scales = policy_noise_scales(self.raw_default_q)
    self.kp_scale = torch.ones(n, 6, device=dev)
    self.kd_scale = torch.ones(n, 6, device=dev)
    self.torque_scale = torch.ones(n, 6, device=dev)
    self.default_offset = torch.zeros(n, 6, device=dev)
    self.friction = torch.full((n,), 0.8, device=dev)
    self.restitution = torch.zeros(n, device=dev)
    self.base_mass = torch.full((n,), 14.0, device=dev)
    self.base_com = torch.zeros(n, 3, device=dev)
    self.torque = torch.zeros(n, 6, device=dev)
    self.base_air_time = torch.zeros(n, device=dev)
    self.last_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    self.contact_filt = self.last_contact.clone()
    self.in_flight = torch.zeros(n, dtype=torch.bool, device=dev)
    self.fail_count = torch.zeros(n, dtype=torch.long, device=dev)
    self.cached_step = -1
    self.cached_noisy_step = -1
    self.cached_noisy_obs = torch.zeros(n, 25, device=dev)
    self._initialize_kinematic_history()

  @property
  def default_q(self) -> torch.Tensor:
    return self.raw_default_q + self.default_offset

  def _raw(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = self.robot.data.data
    root_qadr = self.robot.indexing.free_joint_q_adr
    return (
      data.qpos[:, root_qadr[:3]],
      data.qpos[:, root_qadr[3:7]],
      data.qpos[:, self.q_adr],
    )

  def _initialize_kinematic_history(self) -> None:
    pos, quat, active_q = self._raw()
    self.previous_root_pos = pos.clone()
    self.previous_substep_q = active_q.clone()
    self.substep_vel = torch.zeros_like(active_q)
    self.previous_active_vel = torch.zeros_like(active_q)
    self.root_pos = pos.clone()
    self.root_quat = quat.clone()
    self.active_q = active_q.clone()
    self.active_vel = torch.zeros_like(active_q)
    self.base_lin_vel = torch.zeros_like(pos)
    root_vadr = self.robot.indexing.free_joint_v_adr
    self.base_ang_vel = self.robot.data.data.qvel[:, root_vadr[3:6]].clone()
    self.projected_gravity = quat_rotate_inverse(quat, self.gravity)
    self.dof_acc = torch.zeros_like(active_q)
    self.length, self.angle = virtual_leg_geometry(active_q, motor_zero=self.motor_zero)
    self.last_qpos_version = self.env._sim_step_counter

  def sample_substep_velocity(self, active_q: torch.Tensor, qpos_version: int) -> torch.Tensor:
    if qpos_version == self.last_qpos_version:
      return self.substep_vel
    diff = torch.remainder(active_q - self.previous_substep_q + math.pi, 2 * math.pi) - math.pi
    self.substep_vel.copy_(diff / self.env.physics_dt)
    self.previous_substep_q.copy_(active_q)
    self.last_qpos_version = qpos_version
    return self.substep_vel

  def refresh(self) -> WbrState:
    if self.cached_step == self.env.common_step_counter:
      return self
    pos, quat, active_q = self._raw()
    root_vadr = self.robot.indexing.free_joint_v_adr
    ang_vel = self.robot.data.data.qvel[:, root_vadr[3:6]]
    self.root_pos = pos.clone()
    self.root_quat = quat.clone()
    self.active_q = active_q.clone()
    self.active_vel = self.sample_substep_velocity(active_q, self.env._sim_step_counter).clone()
    self.base_lin_vel = quat_rotate_inverse(quat, (pos - self.previous_root_pos) / POLICY_DT)
    self.base_ang_vel = ang_vel.clone()
    self.projected_gravity = quat_rotate_inverse(quat, self.gravity)
    self.dof_acc = (self.previous_active_vel - self.active_vel) / POLICY_DT
    self.length, self.angle = virtual_leg_geometry(active_q, motor_zero=self.motor_zero)
    self.previous_root_pos.copy_(pos)
    self.previous_active_vel.copy_(self.active_vel)
    self._refresh_contacts()
    self.cached_step = self.env.common_step_counter
    return self

  def _refresh_contacts(self) -> None:
    if self.mode != "jump":
      return
    data = self.env.scene["wheel_contact"].data
    force = data.force
    contact = data.found > 0
    if force is not None:
      # netforce is in world coordinates, with sign set by the primary/secondary
      # ordering. On this flat terrain, |Fz| is the normal load in either ordering.
      # Keep the 1 N threshold and require a matched wheel/terrain contact.
      contact &= force[..., 2].abs() > 1.0
    self.contact_filt, self.in_flight = filtered_flight_state(contact, self.last_contact)
    self.last_contact.copy_(contact)

  def reset(self, env_ids: torch.Tensor) -> None:
    # Refresh non-reset rows before patching reset rows in-place. A single scalar
    # cache remains sufficient because mjlab resets only after rewards have refreshed
    # the terminal step; this guard also makes explicit partial reset robust.
    if self.cached_step != self.env.common_step_counter:
      self.refresh()
    pos, quat, active_q = self._raw()
    root_vadr = self.robot.indexing.free_joint_v_adr
    ang_vel = self.robot.data.data.qvel[:, root_vadr[3:6]]
    length, angle = virtual_leg_geometry(active_q, motor_zero=self.motor_zero)
    self.root_pos[env_ids] = pos[env_ids]
    self.root_quat[env_ids] = quat[env_ids]
    self.active_q[env_ids] = active_q[env_ids]
    self.active_vel[env_ids] = 0.0
    self.base_lin_vel[env_ids] = 0.0
    self.base_ang_vel[env_ids] = ang_vel[env_ids]
    self.projected_gravity[env_ids] = quat_rotate_inverse(quat, self.gravity)[env_ids]
    self.dof_acc[env_ids] = 0.0
    self.length[env_ids] = length[env_ids]
    self.angle[env_ids] = angle[env_ids]
    self.previous_root_pos[env_ids] = pos[env_ids]
    self.previous_substep_q[env_ids] = active_q[env_ids]
    self.substep_vel[env_ids] = 0.0
    self.last_qpos_version = self.env._sim_step_counter
    self.previous_active_vel[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.base_air_time[env_ids] = 0.0
    self.last_contact[env_ids] = False
    self.contact_filt[env_ids] = False
    self.in_flight[env_ids] = False
    self.fail_count[env_ids] = 0
    self.cached_step = self.env.common_step_counter
    self.cached_noisy_step = -1


def get_state(env: ManagerBasedRlEnv, mode: Mode) -> WbrState:
  state = getattr(env, "_wbr_state", None)
  if state is None:
    state = WbrState(env, mode)
    env._wbr_state = state
  elif state.mode != mode:
    raise ValueError(f"environment already initialized for {state.mode}")
  return state


@dataclass(kw_only=True)
class HybridActionCfg(ActionTermCfg):
  mode: Mode

  def build(self, env: ManagerBasedRlEnv) -> HybridAction:
    return HybridAction(self, env)


class HybridAction(ActionTerm):
  cfg: HybridActionCfg

  def __init__(self, cfg: HybridActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.state = get_state(env, cfg.mode)
    self._raw_action = torch.zeros(env.num_envs, 6, device=env.device)
    self._constants = hybrid_constants(self._raw_action, cfg.mode)

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action.copy_(actions.clamp(-100.0, 100.0))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    """Preserve the legacy terminal current action while clearing older history."""
    ids = slice(None) if env_ids is None else env_ids
    # ActionManager clears its buffers immediately before invoking term.reset().
    # Restoring only current reproduces the old post-reset state: current=terminal,
    # previous=previous_previous=0. The next process_action then shifts it once.
    self._env.action_manager._action[ids] = self._raw_action[ids]
    return {}

  def apply_actions(self) -> None:
    data = self._entity.data.data
    q = data.qpos[:, self.state.q_adr]
    qd = self.state.sample_substep_velocity(q, self._env._sim_step_counter - 1)
    torque = hybrid_torque(
      self._raw_action,
      q,
      qd,
      self.state.default_q,
      self.state.kp_scale,
      self.state.kd_scale,
      self.state.torque_scale,
      self.cfg.mode,
      constants=self._constants,
    )
    self.state.torque.copy_(torque)
    self._entity.set_joint_effort_target(torque, joint_ids=self.state.joint_ids)
    self._entity.data.tendon_effort_target.zero_()


@dataclass(kw_only=True)
class WbrCommandCfg(CommandTermCfg):
  mode: Mode
  vx: tuple[float, float]
  yaw: tuple[float, float]
  height: tuple[float, float]

  def build(self, env: ManagerBasedRlEnv) -> WbrCommand:
    return WbrCommand(self, env)


class WbrCommand(CommandTerm):
  cfg: WbrCommandCfg

  def __init__(self, cfg: WbrCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.zeros(env.num_envs, 3, device=env.device)
    self.vx_limit = max(abs(cfg.vx[0]), abs(cfg.vx[1]))

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    low = self._command.new_tensor((-self.vx_limit, self.cfg.yaw[0], self.cfg.height[0]))
    high = self._command.new_tensor((self.vx_limit, self.cfg.yaw[1], self.cfg.height[1]))
    self._command[env_ids] = low + torch.rand(len(env_ids), 3, device=self.device) * (high - low)

  def _update_metrics(self) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass


def command(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("motion")


def policy_observation(env: ManagerBasedRlEnv, mode: Mode, noisy: bool) -> torch.Tensor:
  state = get_state(env, mode).refresh()
  if noisy and state.cached_noisy_step == env.common_step_counter:
    return state.cached_noisy_obs.clone()
  clean = policy_vector(
    state.base_ang_vel,
    state.projected_gravity,
    command(env),
    state.active_q,
    state.default_q,
    state.active_vel,
    env.action_manager.action,
    mode,
    command_scale=state.command_scale,
  )
  if not noisy:
    return clean
  if state.cached_noisy_step != env.common_step_counter:
    state.cached_noisy_obs.copy_(add_policy_noise(clean, scales=state.noise_scales))
    state.cached_noisy_step = env.common_step_counter
  # ObservationManager clips in-place. Keep the shared noisy frame pristine so the
  # legacy history remains un-clipped while the policy output is clipped to +/-100.
  return state.cached_noisy_obs.clone()


def critic_observation(env: ManagerBasedRlEnv, mode: Mode) -> torch.Tensor:
  state = get_state(env, mode).refresh()
  return critic_vector(
    state.base_lin_vel,
    policy_observation(env, mode, noisy=False),
    env.action_manager.prev_action,
    env.action_manager.prev_prev_action,
    state.dof_acc,
    state.root_pos[:, 2],
    state.torque,
    state.base_mass,
    state.base_com,
    state.default_offset,
    state.friction,
    state.restitution,
    mode,
  )


def reset_wbr_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, mode: Mode) -> None:
  get_state(env, mode).reset(resolve_env_ids(env, env_ids))


def reset_root_velocity(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene["robot"]
  velocity = torch.empty(len(ids), 6, device=env.device).uniform_(-0.5, 0.5)
  robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)


@requires_model_fields(
  "body_mass",
  "body_inertia",
  "body_ipos",
  "geom_friction",
  "geom_solref",
  recompute=RecomputeLevel.set_const,
)
def randomize_wbr(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, mode: Mode) -> None:
  """Sample the legacy startup-only domain randomization exactly once."""
  ids = resolve_env_ids(env, env_ids)
  state = get_state(env, mode)
  robot = state.robot
  n = len(ids)
  if mode == "plane":
    fr, rest, mass, inertia, com, gain, dq = (
      (0.6, 1.4),
      (0.6, 1.0),
      (-1, 2),
      (0.9, 1.1),
      0.02,
      (0.95, 1.05),
      0.03,
    )
  else:
    fr, rest, mass, inertia, com, gain, dq = (
      (0.1, 2.0),
      (0.5, 1.0),
      (-2, 3),
      (0.8, 1.2),
      0.05,
      (0.9, 1.1),
      0.05,
    )
  buckets = torch.empty(64, device=env.device).uniform_(*fr)
  friction = buckets[torch.randint(0, 64, (n,), device=env.device)]
  restitution = torch.empty(n, device=env.device).uniform_(*rest)
  mass_add = torch.empty(n, device=env.device).uniform_(*mass)
  state.friction[ids] = friction
  state.restitution[ids] = restitution
  state.base_mass[ids] = 14.0 + mass_add
  state.base_com[ids] = torch.empty(n, 3, device=env.device).uniform_(-com, com)
  state.kp_scale[ids] = torch.empty(n, 6, device=env.device).uniform_(*gain)
  state.kd_scale[ids] = torch.empty(n, 6, device=env.device).uniform_(*gain)
  state.torque_scale[ids] = torch.empty(n, 6, device=env.device).uniform_(*gain)
  state.default_offset[ids] = torch.empty(n, 6, device=env.device).uniform_(-dq, dq)
  robot.data.default_joint_pos[ids[:, None], state.joint_ids] = state.default_q[ids]

  env_grid, body_grid = torch.meshgrid(ids, robot.indexing.body_ids, indexing="ij")
  body_scale = torch.empty(n, robot.num_bodies, device=env.device).uniform_(*inertia)
  base_id = robot.indexing.body_ids[0]
  env.sim.model.body_mass[ids, base_id] += mass_add
  env.sim.model.body_ipos[ids, base_id] += state.base_com[ids]
  env.sim.model.body_mass[env_grid, body_grid] *= body_scale
  env.sim.model.body_inertia[env_grid, body_grid] *= body_scale[..., None]
  geom_grid = robot.indexing.geom_ids[None, :].expand(n, -1)
  env.sim.model.geom_friction[ids[:, None], geom_grid, 0] = friction[:, None]
  # Match coefficient of restitution with the equivalent under-damped ratio.
  log_e = torch.log(restitution.clamp_max(0.9999))
  damping_ratio = -log_e / torch.sqrt(math.pi**2 + log_e.square())
  env.sim.model.geom_solref[ids[:, None], geom_grid, 1] = damping_ratio[:, None]
  # MuJoCo takes the maximum friction coefficient from a contacting pair. Give
  # the shared plane the same per-world value so low-friction samples stay active.
  terrain: Entity = env.scene["terrain"]
  terrain_grid = terrain.indexing.geom_ids[None, :].expand(n, -1)
  env.sim.model.geom_friction[ids[:, None], terrain_grid, 0] = friction[:, None]


def reward_term(
  env: ManagerBasedRlEnv, mode: Mode, name: str, weight: float, clip: float
) -> torch.Tensor:
  """Evaluate one legacy term, then weight/clip before manager dt scaling."""
  state = get_state(env, mode).refresh()
  return weighted_clipped_reward(reward_value(state, name), weight, clip)


def fallen(env: ManagerBasedRlEnv, mode: Mode) -> torch.Tensor:
  s = get_state(env, mode).refresh()
  bad = s.projected_gravity[:, 2] > -0.1
  s.fail_count = update_fall_counter(s.fail_count, bad)
  return s.fail_count > int(1.0 / POLICY_DT)


def legacy_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf > env.max_episode_length


def velocity_curriculum(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
) -> dict[str, torch.Tensor]:
  term: WbrCommand = env.command_manager.get_term("motion")
  if env.common_step_counter % env.max_episode_length != 0:
    return {"max_vx": torch.tensor(term.vx_limit, device=env.device)}
  ids = (
    env_ids
    if isinstance(env_ids, torch.Tensor)
    else torch.arange(env.num_envs, device=env.device)[env_ids]
  )
  sums = env.reward_manager._episode_sums
  if len(ids) and "tracking_lin_vel" in sums:
    lin_ok = sums["tracking_lin_vel"][ids].mean() / env.max_episode_length > 0.7 * env.step_dt
    yaw_ok = sums["tracking_ang_vel"][ids].mean() / env.max_episode_length > 0.7 * 0.8 * env.step_dt
    if lin_ok and yaw_ok:
      term.vx_limit = min(2.5, term.vx_limit + 0.1)
  return {"max_vx": torch.tensor(term.vx_limit, device=env.device)}
