"""Two registered WBR tasks built from one compact mjlab factory."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import push_by_setting_velocity, reset_scene_to_default
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .mdp import HybridActionCfg, WbrCommandCfg
from .robot import PENALIZED_COLLISION_GEOM_NAMES, WHEEL_GEOM_NAMES, get_wbr_robot_cfg

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


def make_env_cfg(mode: mdp.Mode, *, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build plane or jump without duplicating an environment class."""
  is_plane = mode == "plane"
  num_envs = 1 if play else (8192 if is_plane else 4096)
  rewards = PLANE_REWARDS if is_plane else JUMP_REWARDS
  clip = 1.0 if is_plane else 2.5
  observations = {
    "policy": ObservationGroupCfg(
      {
        "frame": ObservationTermCfg(
          func=mdp.policy_observation,
          params={"mode": mode, "noisy": not play},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
    "history": ObservationGroupCfg(
      {
        "frame": ObservationTermCfg(
          func=mdp.policy_observation,
          params={"mode": mode, "noisy": not play},
          history_length=5,
          flatten_history_dim=True,
        )
      },
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      {
        "state": ObservationTermCfg(
          func=mdp.critic_observation,
          params={"mode": mode},
          clip=(-100.0, 100.0),
        )
      }
    ),
  }
  events = {
    "startup_randomization": EventTermCfg(
      func=mdp.randomize_wbr, mode="startup", params={"mode": mode}
    ),
    "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    "reset_velocity": EventTermCfg(func=mdp.reset_root_velocity, mode="reset"),
    "reset_state": EventTermCfg(func=mdp.reset_wbr_state, mode="reset", params={"mode": mode}),
  }
  if play:
    events.pop("startup_randomization")
  elif mode == "jump":
    events["push"] = EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 5.0),
      params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    )
  reward_cfg = {
    name: RewardTermCfg(
      func=mdp.reward_term,
      weight=1.0,
      params={"mode": mode, "name": name, "weight": weight, "clip": clip},
    )
    for name, weight in rewards
  }
  contact_secondary = ContactMatch(mode="geom", pattern="terrain")
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=2.0,
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        env_spacing=2.0,
        geoms=(
          GeomCfg(
            geom_names_expr=("terrain",),
            contype=1,
            conaffinity=2,
            friction=(0.8, 0.005, 0.0001),
          ),
        ),
      ),
      entities={"robot": get_wbr_robot_cfg()},
      sensors=(
        ContactSensorCfg(
          name="wheel_contact",
          primary=ContactMatch(mode="geom", pattern=WHEEL_GEOM_NAMES, entity="robot"),
          secondary=contact_secondary,
          fields=("found", "force"),
          reduce="netforce",
          num_slots=1,
        ),
        ContactSensorCfg(
          name="penalized_contact",
          primary=ContactMatch(mode="geom", pattern=PENALIZED_COLLISION_GEOM_NAMES, entity="robot"),
          secondary=contact_secondary,
          fields=("found", "force"),
          reduce="netforce",
          num_slots=1,
        ),
      ),
    ),
    observations=observations,
    actions={"hybrid": HybridActionCfg(entity_name="robot", mode=mode)},
    commands={
      "motion": WbrCommandCfg(
        mode=mode,
        vx=(-2.0, 2.0) if is_plane else (-2.1, 2.1),
        yaw=(-2.0, 2.0),
        height=(0.10, 0.20) if is_plane else (0.12, 0.15),
        resampling_time_range=(5.0, 5.0) if is_plane else (20.0, 20.0),
      )
    },
    events=events,
    rewards=reward_cfg,
    terminations={
      "fallen": TerminationTermCfg(func=mdp.fallen, params={"mode": mode}),
      "time_out": TerminationTermCfg(func=mdp.legacy_timeout, time_out=True),
    },
    curriculum=(
      {"velocity": CurriculumTermCfg(func=mdp.velocity_curriculum)} if is_plane and not play else {}
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.5,
      elevation=-15.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=200,
      njmax=200,
      mujoco=MujocoCfg(
        timestep=0.001,
        integrator="euler",
        solver="newton",
        iterations=20,
        tolerance=1e-9,
      ),
    ),
    decimation=10,
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )
  return cfg
