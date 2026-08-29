"""Reward weights and formulas shared by both WBR tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .robot import LEG_IDS, POLICY_DT

if TYPE_CHECKING:
  from .mdp import WbrState

PLANE_REWARDS = (
  ("tracking_lin_vel", 1.0),
  ("tracking_lin_vel_enhance", 1.0),
  ("tracking_ang_vel", 1.0),
  ("tracking_ang_vel_enhance", 1.0),
  ("base_height", 1.0),
  ("nominal_state", -1.0),
  ("lin_vel_z", -1.0),
  ("ang_vel_xy", -0.2),
  ("orientation", -100.0),
  ("dof_vel", -5e-5),
  ("dof_acc", -2.5e-7),
  ("torques", -1e-4),
  ("action_rate", -0.01),
  ("action_smooth", -0.01),
  ("collision", -1.0),
  ("dof_pos_limits", -1.0),
)

JUMP_REWARDS = (
  ("tracking_lin_vel", 1.0),
  ("tracking_lin_vel_enhance", 1.0),
  ("tracking_ang_vel", 1.0),
  ("flight", 0.15),
  ("encourage_jump", 1.0),
  ("base_height_flight", 6.0),
  ("leg_tuck", 1.7),
  ("takeoff_extend", 0.5),
  ("line_z", 6.0),
  ("pen_theta_no0", -2.0),
  ("action_rate", -0.04),
  ("torques", -5e-5),
  ("orientation", -25.0),
  ("ang_vel_xy", -0.1),
  ("nominal_state", -1.0),
  ("collision", -1.0),
)


def weighted_clipped_reward(raw: torch.Tensor, weight: float, clip: float) -> torch.Tensor:
  return (raw * weight).clamp(-clip, clip)


def joint_position_limit_penalty(q: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
  """Legacy soft-limit distance for the four leg motors."""
  leg_q = q[:, LEG_IDS]
  leg_limits = limits[:, LEG_IDS]
  below = -(leg_q - leg_limits[..., 0]).clamp(max=0.0)
  above = (leg_q - leg_limits[..., 1]).clamp(min=0.0)
  return (below + above).sum(dim=1)


def reward_value(s: WbrState, name: str) -> torch.Tensor:
  """Evaluate only the requested term; clipping and dt scaling happen outside."""
  env, mode = s.env, s.mode
  match name:
    case "tracking_lin_vel" | "tracking_lin_vel_enhance":
      err = (env.command_manager.get_command("motion")[:, 0] - s.base_lin_vel[:, 0]).square()
      raw = torch.exp(-err / 0.25) if name == "tracking_lin_vel" else torch.exp(-err / 2.5) - 1.0
      return raw * (2.0 if mode == "jump" else 1.0)
    case "tracking_ang_vel" | "tracking_ang_vel_enhance":
      err = (env.command_manager.get_command("motion")[:, 1] - s.base_ang_vel[:, 2]).square()
      return torch.exp(-err / 0.25) if name == "tracking_ang_vel" else torch.exp(-err / 2.5) - 1.0
    case "base_height":
      cmd = env.command_manager.get_command("motion")
      return torch.exp(-(s.root_pos[:, 2] - cmd[:, 2]).square() / 0.001)
    case "nominal_state":
      return (s.angle[:, 0] - s.angle[:, 1]).square() + (
        10.0 * (s.length[:, 0] - s.length[:, 1]).square() if mode == "jump" else 0.0
      )
    case "lin_vel_z":
      return s.base_lin_vel[:, 2].square()
    case "ang_vel_xy":
      return s.base_ang_vel[:, :2].square().sum(1)
    case "orientation":
      return s.projected_gravity[:, :2].square().sum(1)
    case "dof_vel":
      return s.active_vel[:, LEG_IDS].square().sum(1)
    case "dof_acc":
      return s.dof_acc.square().sum(1)
    case "torques":
      return s.torque.square().sum(1)
    case "action_rate":
      return (env.action_manager.action - env.action_manager.prev_action).square().sum(1)
    case "action_smooth":
      actions = env.action_manager
      return (
        (
          actions.action[:, LEG_IDS]
          - 2.0 * actions.prev_action[:, LEG_IDS]
          + actions.prev_prev_action[:, LEG_IDS]
        )
        .square()
        .sum(1)
      )
    case "dof_pos_limits":
      return joint_position_limit_penalty(
        s.active_q, s.robot.data.soft_joint_pos_limits[:, s.joint_ids]
      )
    case "flight":
      return s.in_flight.float()
    case "base_height_flight":
      return torch.exp(-torch.abs(s.root_pos[:, 2] - 0.65) * 6.0) * s.in_flight
    case "leg_tuck":
      return torch.exp(-torch.abs(s.length - 0.16).sum(1) * 4.0) * s.in_flight
    case "takeoff_extend":
      vertical = s.robot.data.data.qvel[:, s.robot.indexing.free_joint_v_adr[2]]
      return torch.exp(-torch.abs(s.length - 0.31).sum(1) * 4.0) * (
        s.contact_filt.any(1) & (vertical > 0.15)
      )
    case "line_z":
      vertical = s.robot.data.data.qvel[:, s.robot.indexing.free_joint_v_adr[2]]
      return vertical.clamp_min(0.0) * s.in_flight
    case "pen_theta_no0":
      return s.angle.square().sum(1)
    case "collision":
      if mode == "plane":
        return torch.zeros(env.num_envs, device=env.device)
      force = env.scene["penalized_contact"].data.force
      return (torch.linalg.vector_norm(force, dim=-1) > 0.1).float().sum(1)
    case "encourage_jump":
      # Accumulate the height-weighted flight interval, pay it once on landing,
      # and clear it on the ground. Retaining it on the ground rewards standing.
      vertical = s.robot.data.data.qvel[:, s.robot.indexing.free_joint_v_adr[2]]
      first_contact = (s.base_air_time > 0.0) & ~s.in_flight
      s.base_air_time += POLICY_DT * s.root_pos[:, 2].clamp(0.0, 0.5)
      raw = (s.base_air_time - 5e-5) * first_contact * 0.15 + vertical.clamp_min(0.0) * 0.15
      s.base_air_time *= s.in_flight
      return raw
    case _:
      raise KeyError(name)
