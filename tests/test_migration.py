from __future__ import annotations

import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wbr_mjlab import JUMP_TASK_ID, PLANE_TASK_ID
from wbr_mjlab.mdp import (
  WbrState,
  add_policy_noise,
  critic_vector,
  filtered_flight_state,
  hybrid_constants,
  hybrid_torque,
  joint_position_limit_penalty,
  policy_noise_scales,
  policy_vector,
  reward_term,
  update_fall_counter,
  virtual_leg_geometry,
  weighted_clipped_reward,
)
from wbr_mjlab.rewards import reward_value
from wbr_mjlab.rl import SequencePolicy, SequencePPO, export_onnx, sequence_runner_cfg
from wbr_mjlab.robot import (
  ACTIVE_JOINT_NAMES,
  HOME_ACTIVE_JOINT_POS,
  HOME_JOINT_POS,
  IMU_OFFSET,
  MOTOR_ACTUATOR_NAMES,
  MOTOR_ZERO_RAD,
  PENALIZED_COLLISION_GEOM_NAMES,
  ROBOT_COLLISION_GEOM_NAMES,
  TORQUE_LIMITS,
  WBR_XML_PATH,
  WHEEL_RADIUS,
  load_wbr_spec,
)
from wbr_mjlab.task import JUMP_REWARDS, PLANE_REWARDS, make_env_cfg


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str | None:
  return mujoco.mj_id2name(model, obj, index)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return load_wbr_spec().compile()


def test_mjcf_topology_and_vendored_assets(model: mujoco.MjModel) -> None:
  assert (model.nq, model.nv, model.nu, model.nbody, model.njnt) == (21, 20, 8, 16, 15)
  assert (model.neq, model.ntendon, model.ngeom, model.nmesh) == (4, 2, 15, 15)
  assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
  assert set(ROBOT_COLLISION_GEOM_NAMES) == {model.geom(i).name for i in range(model.ngeom)}
  assert len(PENALIZED_COLLISION_GEOM_NAMES) == 13
  assert np.all(model.geom_contype == 2) and np.all(model.geom_conaffinity == 1)
  manifest = json.loads((WBR_XML_PATH.parent / "UPSTREAM.json").read_text())
  mesh_files = {
    "meshes/" + mesh.attrib["file"] for mesh in ET.parse(WBR_XML_PATH).findall("./asset/mesh")
  }
  assert set(manifest["sha256"]) == {"mjmodel.xml", *mesh_files}
  for path, expected in manifest["sha256"].items():
    assert hashlib.sha256((WBR_XML_PATH.parent / path).read_bytes()).hexdigest() == expected
  # The adapter must not alter source dynamics, geometry, or sensor positions.
  upstream = mujoco.MjModel.from_xml_path(str(WBR_XML_PATH))
  for field in (
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "jnt_axis",
    "jnt_range",
    "jnt_actfrcrange",
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
    "geom_type",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "mesh_vert",
    "mesh_face",
    "actuator_forcerange",
    "actuator_biasprm",
    "site_pos",
  ):
    np.testing.assert_array_equal(getattr(model, field), getattr(upstream, field))
  spec = load_wbr_spec()
  spec.worldbody.add_geom(
    name="terrain",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.1),
    contype=1,
    conaffinity=2,
  )
  floor_model = spec.compile()
  data = mujoco.MjData(floor_model)
  mujoco.mj_resetDataKeyframe(floor_model, data, 0)
  mujoco.mj_forward(floor_model, data)
  assert data.ncon == 0  # No mesh penetration at reset, including closed-chain pivots.
  assert np.max(np.abs(data.efc_pos[:12])) < 1e-7
  data.qpos[2] -= 0.002
  mujoco.mj_forward(floor_model, data)
  touching = {floor_model.geom(gid).name for c in data.contact for gid in c.geom}
  assert {"terrain", "collision_lwheel", "collision_rwheel"} <= touching


def test_motor_order_axes_limits_and_home(model: mujoco.MjModel) -> None:
  actuators = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
  assert tuple(actuators[:6]) == MOTOR_ACTUATOR_NAMES
  expected_axes = ((0, -1, 0), (0, -1, 0), (0, -1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0))
  for name, axis, limit in zip(ACTIVE_JOINT_NAMES, expected_axes, TORQUE_LIMITS, strict=True):
    jid = model.joint(name).id
    np.testing.assert_allclose(model.jnt_axis[jid], axis)
    # Upstream ljoint4 has a 40 Nm joint cap, but its motor is limited to 20 Nm.
    assert model.jnt_actfrcrange[jid, 0] <= -limit
    assert model.jnt_actfrcrange[jid, 1] >= limit
    np.testing.assert_allclose(model.actuator(name + "_actuator").forcerange, (-limit, limit))
  assert model.nkey == 1
  assert model.key("home").qpos[2] == pytest.approx(0.175)
  home = [model.key("home").qpos[model.jnt_qposadr[model.joint(n).id]] for n in ACTIVE_JOINT_NAMES]
  np.testing.assert_allclose(home, HOME_ACTIVE_JOINT_POS)
  complete_home = {
    _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid): model.key("home").qpos[model.jnt_qposadr[jid]]
    for jid in range(1, model.njnt)
  }
  assert complete_home == pytest.approx(HOME_JOINT_POS)
  np.testing.assert_allclose(model.site("imu").pos, IMU_OFFSET)


def test_closed_chains_tendons_and_passive_springs(model: mujoco.MjModel) -> None:
  assert [_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(4)] == [
    "Left_loop1",
    "Left_loop2",
    "Right_loop1",
    "Right_loop2",
  ]
  np.testing.assert_allclose(model.tendon_range, ((0.12, 0.195), (0.12, 0.195)))
  np.testing.assert_allclose(model.actuator_forcerange[:6], [(-20, 20), (-20, 20), (-5.2, 5.2)] * 2)
  np.testing.assert_allclose(model.actuator_forcerange[6:], ((250, 385), (250, 385)))
  np.testing.assert_allclose(model.actuator_biasprm[6:, :3], ((34, 1800, -4),) * 2)
  np.testing.assert_allclose(model.actuator_gainprm[6:, 0], 0.0)


def test_policy_and_critic_exact_layout() -> None:
  n = 2
  tensors = [
    torch.arange(n * d, dtype=torch.float32).reshape(n, d) / 10 for d in (3, 3, 3, 6, 6, 6)
  ]
  ang, gravity, cmd, q, qd, action = tensors
  default = q - 0.1
  policy = policy_vector(ang, gravity, cmd, q, default, qd, action, "plane")
  assert policy.shape == (n, 25)
  torch.testing.assert_close(policy[:, :3], ang * 0.25)
  torch.testing.assert_close(policy[:, 3:6], gravity)
  torch.testing.assert_close(policy[:, 6:9], cmd * torch.tensor((2.0, 0.25, 5.0)))
  torch.testing.assert_close(policy[:, 9:13], (q - default)[:, (0, 1, 3, 4)])
  torch.testing.assert_close(policy[:, 13:19], qd * 0.05)
  torch.testing.assert_close(policy[:, 19:], action)
  critic = critic_vector(
    ang,
    policy,
    action + 1,
    action + 2,
    qd,
    torch.tensor((0.2, 0.8)),
    action,
    torch.tensor((14.0, 16.0)),
    gravity,
    q - default,
    torch.tensor((0.6, 1.4)),
    torch.tensor((0.7, 0.9)),
    "plane",
  )
  assert critic.shape == (n, 141)
  torch.testing.assert_close(critic[:, :3], ang * 2.0)
  torch.testing.assert_close(critic[:, 3:28], policy)
  torch.testing.assert_close(critic[:, 28:34], action + 1)
  torch.testing.assert_close(critic[:, 34:40], action + 2)
  torch.testing.assert_close(critic[:, 40:46], qd * 0.0025)
  assert torch.all(critic[0, 46:123] == -1.5)
  assert torch.all(critic[1, 46:123] == 1.5)


def test_noise_and_frame_major_history() -> None:
  torch.manual_seed(4)
  obs = torch.zeros(128, 25)
  noisy = add_policy_noise(obs)
  assert torch.all(noisy[:, 6:9] == 0)
  assert torch.all(noisy[:, 19:25] == 0)
  assert noisy[:, :6].abs().max() <= 0.05
  assert noisy[:, 9:13].abs().max() <= 0.02
  assert noisy[:, 13:19].abs().max() <= 0.075
  frames = [torch.full((2, 25), float(i)) for i in range(5)]
  history = torch.stack(frames, dim=1).flatten(1)
  assert history.shape == (2, 125)
  assert history[0, ::25].tolist() == [0, 1, 2, 3, 4]


def test_jump_filter_and_continuous_fall_counter() -> None:
  previous = torch.tensor(((True, False), (False, False)))
  current = torch.tensor(((False, False), (False, False)))
  filtered, flight = filtered_flight_state(current, previous)
  assert filtered.tolist() == [[True, False], [False, False]]
  assert flight.tolist() == [False, True]
  count = torch.tensor((99, 100, 50))
  count = update_fall_counter(count, torch.tensor((True, True, False)))
  assert count.tolist() == [100, 101, 0]
  assert (count > 100).tolist() == [False, True, False]
  raw = torch.tensor((-3.0, 0.25, 3.0))
  torch.testing.assert_close(
    weighted_clipped_reward(raw, weight=2.0, clip=1.0),
    torch.tensor((-1.0, 0.5, 1.0)),
  )
  q = torch.tensor(((0.0, -2.0, 99.0, 3.0, 0.0, 99.0),))
  limits = torch.tensor([[(-1.0, 1.0)] * 6])
  torch.testing.assert_close(joint_position_limit_penalty(q, limits), torch.tensor((3.0,)))


def test_jump_contact_force_sign_threshold_and_transitions() -> None:
  # Exercise the actual state update, including signed forces, stale unmatched
  # forces, horizontal-only forces and the one-frame contact dropout filter.
  data = SimpleNamespace(
    found=torch.tensor(((1, 0), (0, 1), (1, 1), (0, 0), (1, 1))),
    force=torch.tensor(
      (
        ((0.0, 0.0, -50.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 50.0)),
        ((50.0, 0.0, 0.0), (0.0, 50.0, 0.0)),
        ((0.0, 0.0, -50.0), (0.0, 0.0, 50.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
      )
    ),
  )
  state = SimpleNamespace(
    mode="jump",
    env=SimpleNamespace(scene={"wheel_contact": SimpleNamespace(data=data)}),
    last_contact=torch.zeros(5, 2, dtype=torch.bool),
  )
  WbrState._refresh_contacts(state)
  assert state.last_contact.tolist() == [
    [True, False],
    [False, True],
    [False, False],
    [False, False],
    [False, False],
  ]
  assert state.in_flight.tolist() == [False, False, True, True, True]
  data.found.zero_()
  data.force.zero_()
  WbrState._refresh_contacts(state)
  assert state.in_flight.tolist() == [False, False, True, True, True]
  WbrState._refresh_contacts(state)
  assert state.in_flight.all()
  data.found[0, 1] = 1
  data.force[0, 1, 2] = -50.0
  WbrState._refresh_contacts(state)
  assert state.in_flight.tolist() == [False, True, True, True, True]
  # Sensors without a force field still fall back to their contact matches.
  data.force = None
  data.found[1, 0] = 1
  WbrState._refresh_contacts(state)
  assert state.in_flight.tolist() == [False, False, True, True, True]


def test_jump_air_time_pays_once_on_landing() -> None:
  state = SimpleNamespace(
    env=None,
    mode="jump",
    base_air_time=torch.zeros(1),
    root_pos=torch.tensor(((0.0, 0.0, 0.2),)),
    in_flight=torch.tensor((False,)),
    robot=SimpleNamespace(
      indexing=SimpleNamespace(free_joint_v_adr=[0, 1, 2]),
      data=SimpleNamespace(data=SimpleNamespace(qvel=torch.zeros(1, 3))),
    ),
  )
  for _ in range(3):
    assert reward_value(state, "encourage_jump").item() == 0.0
    assert state.base_air_time.item() == 0.0
  state.in_flight.fill_(True)
  for _ in range(2):
    assert reward_value(state, "encourage_jump").item() == 0.0
  assert state.base_air_time.item() == pytest.approx(0.004)
  state.in_flight.fill_(False)
  assert reward_value(state, "encourage_jump").item() == pytest.approx((0.006 - 5e-5) * 0.15)
  assert state.base_air_time.item() == 0.0
  assert reward_value(state, "encourage_jump").item() == 0.0


@pytest.mark.parametrize("mode", ("plane", "jump"))
def test_hybrid_action_and_clipping(mode: str) -> None:
  action = torch.tensor(((1.0, -1.0, 0.5, 1.0, -1.0, -0.5),))
  q = torch.zeros_like(action)
  qd = torch.zeros_like(action)
  default = torch.zeros_like(action)
  scale = torch.ones_like(action)
  torque = hybrid_torque(action, q, qd, default, scale, scale, scale, mode)
  leg_kp = 20.0 if mode == "plane" else 6.0
  torch.testing.assert_close(torque[:, (0, 1, 3, 4)], action[:, (0, 1, 3, 4)] * 0.5 * leg_kp)
  torch.testing.assert_close(torque[:, (2, 5)], action[:, (2, 5)] * 2.0)
  doubled = hybrid_torque(action, q, qd, default, scale, scale, scale * 2.0, mode)
  torch.testing.assert_close(doubled, torque * 2.0)
  huge = hybrid_torque(action * 1e4, q, qd, default, scale, scale, scale, mode)
  torch.testing.assert_close(huge.abs(), torch.tensor(TORQUE_LIMITS)[None])
  # Cached GPU-path constants must retain the same control law for randomized
  # positions, velocities, gain randomization, and both torque saturation signs.
  torch.manual_seed(19)
  action, q, qd, default = (torch.randn(32, 6) for _ in range(4))
  kp, kd, strength = (torch.rand(32, 6) + 0.5 for _ in range(3))
  expected = torch.empty_like(action)
  for i in range(6):
    if i in (2, 5):
      expected[:, i] = 0.2 * kd[:, i] * (10.0 * action[:, i] - qd[:, i])
    else:
      expected[:, i] = (
        leg_kp * kp[:, i] * (default[:, i] + 0.5 * action[:, i] - q[:, i])
        - (1.0 if mode == "plane" else 0.5) * kd[:, i] * qd[:, i]
      )
  expected *= strength
  limits = torch.tensor(TORQUE_LIMITS)
  expected = expected.clamp(-limits, limits)
  actual = hybrid_torque(
    action,
    q,
    qd,
    default,
    kp,
    kd,
    strength,
    mode,
    constants=hybrid_constants(action, mode),
  )
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wbr_control_geometry_is_mirrored_and_continuous(model: mujoco.MjModel) -> None:
  home = torch.tensor(HOME_ACTIVE_JOINT_POS)[None]
  length, angle = virtual_leg_geometry(home)
  torch.testing.assert_close(length[:, 0], length[:, 1], atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(angle[:, 0], angle[:, 1], atol=1e-6, rtol=1e-6)
  perturbed = home.repeat(101, 1)
  perturbed[:, 0] += torch.linspace(-1e-3, 1e-3, 101)
  lengths, angles = virtual_leg_geometry(perturbed)
  assert torch.isfinite(lengths).all() and torch.isfinite(angles).all()
  assert torch.diff(lengths[:, 0]).abs().max() < 1e-3
  data = mujoco.MjData(model)
  data.qpos[:] = model.key("home").qpos
  mujoco.mj_forward(model, data)
  left, right = data.xpos[model.body("lwlink").id], data.xpos[model.body("rwlink").id]
  left_hip = data.xpos[model.body("llink1").id]
  wheel_delta = left - left_hip
  wheel_length = np.linalg.norm(wheel_delta[[0, 2]])
  wheel_angle = math.atan2(wheel_delta[0], -wheel_delta[2])
  assert length[0, 0].item() == pytest.approx(wheel_length, abs=0.01)
  assert angle[0, 0].item() == pytest.approx(wheel_angle, abs=0.05)
  np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=2e-6)
  assert left[1] == pytest.approx(-right[1], abs=2e-6)
  np.testing.assert_allclose(MOTOR_ZERO_RAD, (-0.06, -0.20, 0, 0.06, 0.20, 0))
  assert WHEEL_RADIUS == pytest.approx(0.060)


def test_task_configuration_and_reward_table() -> None:
  plane, jump = make_env_cfg("plane"), make_env_cfg("jump")
  assert (plane.scene.num_envs, jump.scene.num_envs) == (8192, 4096)
  assert (plane.sim.nconmax, plane.sim.njmax) == (64, 128)
  assert (jump.sim.nconmax, jump.sim.njmax) == (64, 128)
  assert plane.episode_length_s == jump.episode_length_s == 20.0
  assert plane.decimation == jump.decimation == 10
  assert plane.sim.mujoco.timestep == jump.sim.mujoco.timestep == 0.001
  assert plane.sim.mujoco.solver == "newton" and plane.sim.mujoco.iterations == 20
  assert plane.sim.mujoco.tolerance == 1e-9
  assert list(plane.observations) == ["policy", "history", "critic"]
  assert plane.observations["history"].terms["frame"].history_length == 5
  assert plane.observations["policy"].terms["frame"].clip == (-100.0, 100.0)
  assert plane.observations["history"].terms["frame"].clip is None
  assert plane.observations["critic"].terms["state"].clip == (-100.0, 100.0)
  assert dict(PLANE_REWARDS) == {
    "tracking_lin_vel": 1,
    "tracking_lin_vel_enhance": 1,
    "tracking_ang_vel": 1,
    "tracking_ang_vel_enhance": 1,
    "base_height": 1,
    "nominal_state": -1,
    "lin_vel_z": -1,
    "ang_vel_xy": -0.2,
    "orientation": -100,
    "dof_vel": -5e-5,
    "dof_acc": -2.5e-7,
    "torques": -1e-4,
    "action_rate": -0.01,
    "action_smooth": -0.01,
    "collision": -1,
    "dof_pos_limits": -1,
  }
  assert len(JUMP_REWARDS) == 16
  assert "velocity" in plane.curriculum and not jump.curriculum
  assert "push" not in plane.events and "push" in jump.events
  assert asdict(sequence_runner_cfg("plane"))["algorithm"]["desired_kl"] == 0.005
  assert asdict(sequence_runner_cfg("plane"))["algorithm"]["num_learning_epochs"] == 3
  assert sequence_runner_cfg("plane").seed == 1


@pytest.mark.parametrize("mode", ("plane", "jump"))
def test_reward_formulas_clipping_and_air_time(mode: str) -> None:
  """Fixed grounded/airborne states cover every configured reward branch."""
  cmd = torch.tensor(((0.0, 0.0, 0.2), (0.0, 0.0, 0.2)))
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    command_manager=SimpleNamespace(get_command=lambda _: cmd),
    action_manager=SimpleNamespace(
      action=torch.ones(2, 6), prev_action=torch.zeros(2, 6), prev_prev_action=torch.ones(2, 6)
    ),
    scene={
      "penalized_contact": SimpleNamespace(
        data=SimpleNamespace(
          force=torch.tensor(
            (((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
          )
        )
      )
    },
  )
  state = SimpleNamespace(
    env=env,
    mode=mode,
    joint_ids=torch.arange(6),
    robot=SimpleNamespace(
      indexing=SimpleNamespace(free_joint_v_adr=[0, 1, 2, 3, 4, 5]),
      data=SimpleNamespace(
        data=SimpleNamespace(qvel=torch.tensor(((0.0, 0.0, 0.2), (0.0, 0.0, 0.4)))),
        soft_joint_pos_limits=torch.tensor([[(-0.1, 0.1)] * 6] * 2),
      ),
    ),
    base_lin_vel=torch.tensor(((0.5, 0.0, -0.5), (0.0, 0.0, 0.0))),
    base_ang_vel=torch.tensor(((0.0, 0.0, 0.5), (0.0, 0.0, 0.0))),
    root_pos=torch.tensor(((0.0, 0.0, 0.2), (0.0, 0.0, 0.65))),
    projected_gravity=torch.tensor(((0.6, 0.0, -0.8), (0.0, 0.0, -1.0))),
    active_q=torch.full((2, 6), 0.2),
    active_vel=torch.ones(2, 6),
    dof_acc=torch.full((2, 6), 2.0),
    torque=torch.full((2, 6), 3.0),
    angle=torch.tensor(((0.1, -0.1), (0.0, 0.0))),
    length=torch.tensor(((0.31, 0.26), (0.16, 0.16))),
    in_flight=torch.tensor((False, True)),
    contact_filt=torch.tensor(((True, False), (False, False))),
    base_air_time=torch.tensor((0.5, 0.0)),
  )
  state.refresh = lambda: state
  env._wbr_state = state
  scale = 2 if mode == "jump" else 1
  expected = {
    "tracking_lin_vel": (math.exp(-1) * scale, scale),
    "tracking_lin_vel_enhance": ((math.exp(-0.1) - 1) * scale, 0.0),
    "tracking_ang_vel": (math.exp(-1), 1.0),
    "tracking_ang_vel_enhance": (math.exp(-0.1) - 1, 0.0),
    "base_height": (1.0, math.exp(-202.5)),
    "nominal_state": (0.065 if mode == "jump" else 0.04, 0.0),
    "lin_vel_z": (0.25, 0.0),
    "ang_vel_xy": (0.0, 0.0),
    "orientation": (0.36, 0.0),
    "dof_vel": (4.0, 4.0),
    "dof_acc": (24.0, 24.0),
    "torques": (54.0, 54.0),
    "action_rate": (6.0, 6.0),
    "action_smooth": (16.0, 16.0),
    "dof_pos_limits": (0.4, 0.4),
    "flight": (0.0, 1.0),
    "base_height_flight": (0.0, 1.0),
    "leg_tuck": (0.0, 1.0),
    "takeoff_extend": (math.exp(-0.2), 0.0),
    "line_z": (0.0, 0.4),
    "pen_theta_no0": (0.02, 0.0),
    "collision": (0.0, 0.0) if mode == "plane" else (1.0, 2.0),
    "encourage_jump": ((0.502 - 5e-5) * 0.15 + 0.2 * 0.15, 0.4 * 0.15),
  }
  for name, cfg in make_env_cfg(mode).rewards.items():
    params = cfg.params
    raw = torch.tensor(expected[name])
    target = (raw * params["weight"]).clamp(-params["clip"], params["clip"])
    torch.testing.assert_close(reward_term(env, **params), target)
  torch.testing.assert_close(
    state.base_air_time, torch.tensor((0.0, 0.005) if mode == "jump" else (0.5, 0.0))
  )


class _FakeEnv:
  num_envs = 8
  num_actions = 6


def _random_obs(n: int = 8) -> TensorDict:
  return TensorDict(
    {
      "policy": torch.randn(n, 25),
      "history": torch.randn(n, 125),
      "critic": torch.randn(n, 141),
    },
    batch_size=[n],
  )


def _filled_algorithm() -> SequencePPO:
  cfg = asdict(sequence_runner_cfg("plane"))
  cfg["num_steps_per_env"] = 2
  cfg["algorithm"]["num_learning_epochs"] = 1
  cfg["algorithm"]["num_mini_batches"] = 1
  obs = _random_obs()
  alg = SequencePPO.construct_algorithm(obs, _FakeEnv(), cfg, "cpu")
  for _ in range(2):
    alg.act(obs)
    alg.process_env_step(obs, torch.randn(8), torch.zeros(8, dtype=torch.long), {})
    obs = _random_obs()
  alg.compute_returns(obs)
  return alg


def test_encoder_is_frozen_for_ppo_and_updated_by_auxiliary() -> None:
  torch.manual_seed(10)
  alg = _filled_algorithm()
  initial = [p.detach().clone() for p in alg.policy.encoder.parameters()]
  alg.ppo_update()
  for before, after in zip(initial, alg.policy.encoder.parameters(), strict=True):
    torch.testing.assert_close(before, after)
  alg.auxiliary_update()
  assert any(
    not torch.equal(before, after)
    for before, after in zip(initial, alg.policy.encoder.parameters(), strict=True)
  )


def test_ppo_reuses_one_history_encoding_per_batch() -> None:
  torch.manual_seed(11)
  alg = _filled_algorithm()
  calls = 0

  def count_calls(_module, _args, _output) -> None:
    nonlocal calls
    calls += 1

  handle = alg.policy.encoder.register_forward_hook(count_calls)
  try:
    alg.ppo_update()
  finally:
    handle.remove()
  assert calls == alg.cfg.num_learning_epochs * alg.cfg.num_mini_batches


def test_checkpoint_restores_models_optimizers_and_rng(tmp_path: Path) -> None:
  torch.manual_seed(12)
  np.random.seed(12)
  random.seed(12)
  alg = _filled_algorithm()
  alg.update()
  checkpoint = tmp_path / "model.pt"
  torch.save(alg.save(), checkpoint)
  torch_expected, np_expected, py_expected = torch.rand(3), np.random.rand(3), random.random()
  restored = _filled_algorithm()
  restored.load(torch.load(checkpoint, weights_only=False))
  torch.testing.assert_close(torch.rand(3), torch_expected)
  np.testing.assert_allclose(np.random.rand(3), np_expected)
  assert random.random() == py_expected
  obs = _random_obs()
  torch.testing.assert_close(restored.policy(obs), alg.policy(obs))
  assert (
    restored.optimizer.state_dict()["param_groups"] == alg.optimizer.state_dict()["param_groups"]
  )

  fixed = _random_obs()

  def continue_once(item: SequencePPO) -> None:
    actor_loss = item.policy(fixed).square().mean()
    critic_loss = item._value(fixed).square().mean()
    item.optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    item.optimizer.step()
    encoder_loss = item.policy.encoder(fixed["history"]).square().mean()
    item.encoder_optimizer.zero_grad()
    encoder_loss.backward()
    item.encoder_optimizer.step()

  continue_once(alg)
  continue_once(restored)
  for expected, actual in zip(alg.policy.parameters(), restored.policy.parameters(), strict=True):
    torch.testing.assert_close(expected, actual)
  for expected, actual in zip(alg.critic.parameters(), restored.critic.parameters(), strict=True):
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("mode", ("plane", "jump"))
def test_onnx_contract_matches_torch(tmp_path: Path, mode: str) -> None:
  ort = pytest.importorskip("onnxruntime")
  from wbr_mjlab.sim2sim import NativeRunner, OnnxPolicy

  torch.manual_seed(7)
  policy = SequencePolicy().eval()
  output = tmp_path / "policy.onnx"
  export_onnx(policy, output, mode=mode)
  session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
  assert [(x.name, x.shape[1:]) for x in session.get_inputs()] == [
    ("obs", [25]),
    ("obs_history", [125]),
  ]
  assert session.get_outputs()[0].name == "actions"
  obs, history = torch.randn(4, 25), torch.randn(4, 125)
  expected = policy.as_onnx()(obs, history).detach().numpy()
  actual = session.run(None, {"obs": obs.numpy(), "obs_history": history.numpy()})[0]
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
  deployed = OnnxPolicy(output, mode)
  np.testing.assert_allclose(deployed(obs[:1].numpy(), history[:1].numpy()), expected[0], atol=1e-5)
  with pytest.raises(ValueError, match="contract mismatch"):
    OnnxPolicy(output, "jump" if mode == "plane" else "plane")
  with pytest.raises(FloatingPointError, match="Non-finite policy input"):
    deployed(np.full((1, 25), np.nan, dtype=np.float32), history[:1].numpy())
  runner = NativeRunner(deployed, mode)
  for _ in range(20):
    runner.step(np.array([.2, .1, .15], dtype=np.float32))
  assert runner.data.time == pytest.approx(.2)
  assert np.isfinite(runner.data.qpos).all()


@pytest.mark.parametrize("mode,task_id", (("plane", PLANE_TASK_ID), ("jump", JUMP_TASK_ID)))
def test_cpu_environment_reset_and_rollout(mode: str, task_id: str) -> None:
  del task_id
  from mjlab.envs import ManagerBasedRlEnv

  cfg = make_env_cfg(mode, play=False)
  cfg.scene.num_envs = 4
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    obs, _ = env.reset(seed=3)
    assert obs["policy"].shape == (4, 25)
    assert obs["history"].shape == (4, 125)
    assert obs["critic"].shape == (4, 141)
    assert all(torch.isfinite(value).all() for value in obs.values())
    history = obs["history"].reshape(4, 5, 25)
    torch.testing.assert_close(history[:, :1].expand_as(history), history)
    state = env._wbr_state
    if mode == "plane":
      assert state.friction.min() >= 0.6 and state.friction.max() <= 1.4
      assert state.restitution.min() >= 0.6 and state.restitution.max() <= 1.0
      assert state.default_offset.abs().max() <= 0.03
    else:
      assert state.friction.min() >= 0.1 and state.friction.max() <= 2.0
      assert state.restitution.min() >= 0.5 and state.restitution.max() <= 1.0
      assert state.default_offset.abs().max() <= 0.05
    before = env._sim_step_counter
    obs, reward, _, _, _ = env.step(torch.zeros(4, 6))
    assert env._sim_step_counter - before == 10
    assert torch.isfinite(reward).all()
    torch.testing.assert_close(obs["history"][:, -25:], obs["policy"])
    # Policy and history reuse one noisy frame; clipping the returned policy must
    # never mutate that cache, and cached noise constants must match the reference.
    from wbr_mjlab.mdp import policy_observation

    torch.testing.assert_close(state.noise_scales, policy_noise_scales(obs["policy"]))
    cached = state.cached_noisy_obs.clone()
    returned = policy_observation(env, mode, noisy=True)
    returned.zero_()
    torch.testing.assert_close(policy_observation(env, mode, noisy=True), cached)
    obs, *_ = env.step(torch.randn(4, 6) * 0.5)
    assert all(torch.isfinite(value).all() for value in obs.values())
  finally:
    env.close()


def test_jump_contact_and_flight_rewards_with_real_mesh_model() -> None:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = make_env_cfg("jump", play=True)
  cfg.scene.num_envs = 4
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset(seed=1)
    state = env._wbr_state
    pos, quat, _ = state._raw()
    root = torch.cat((pos, quat, torch.zeros(4, 6)), dim=1)

    def place(height: float) -> None:
      # Controlled sensor snapshots isolate contact detection from policy motion.
      root[:, 2] = height
      state.robot.write_root_state_to_sim(root)
      env.sim.forward()
      env.sim.sense()
      env.scene.update(dt=0.0)
      state.cached_step = -1
      state.refresh()

    place(0.173)  # About 1.2 mm of wheel penetration ensures a measurable load.
    contact = env.scene["wheel_contact"].data
    assert (contact.found > 0).all()
    assert (contact.force[..., 2].abs() > 1.0).all()
    assert state.contact_filt.all() and not state.in_flight.any()
    for name in ("flight", "base_height_flight", "leg_tuck", "line_z"):
      assert not reward_value(state, name).any()

    place(0.675)
    assert not env.scene["wheel_contact"].data.found.any()
    assert not state.in_flight.any()  # Retain the existing one-frame dropout filter.
    place(0.675)
    assert state.in_flight.all()
    assert (reward_value(state, "flight") == 1).all()
    assert (reward_value(state, "base_height_flight") > 0).all()
    reward_value(state, "encourage_jump")
    assert (state.base_air_time > 0).all()

    place(0.173)
    assert not state.in_flight.any()  # Landing clears flight immediately.
    assert (reward_value(state, "encourage_jump") > 0).all()
    assert not state.base_air_time.any()
    assert not reward_value(state, "encourage_jump").any()
    # A new episode must not inherit flight or accumulated landing bonuses.
    place(0.675)
    place(0.675)
    reward_value(state, "encourage_jump")
    env.reset(env_ids=torch.arange(4))
    assert not state.base_air_time.any()
    assert not state.last_contact.any()
    assert not state.in_flight.any()
  finally:
    env.close()
