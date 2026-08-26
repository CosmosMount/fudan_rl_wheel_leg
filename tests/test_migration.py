from __future__ import annotations

import math
import random
import re
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wbr_mjlab import JUMP_TASK_ID, PLANE_TASK_ID
from wbr_mjlab.mdp import (
  add_policy_noise,
  critic_vector,
  filtered_flight_state,
  hybrid_torque,
  joint_position_limit_penalty,
  policy_vector,
  update_fall_counter,
  virtual_leg_geometry,
  weighted_clipped_reward,
)
from wbr_mjlab.rl import SequencePolicy, SequencePPO, export_onnx, sequence_runner_cfg
from wbr_mjlab.robot import (
  ACTIVE_JOINT_NAMES,
  HOME_ACTIVE_JOINT_POS,
  HOME_JOINT_POS,
  IMU_OFFSET,
  MOTOR_ACTUATOR_NAMES,
  MOTOR_ZERO_RAD,
  TORQUE_LIMITS,
  WBR_XML_PATH,
  WHEEL_RADIUS,
)
from wbr_mjlab.task import JUMP_REWARDS, PLANE_REWARDS, make_env_cfg


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str | None:
  return mujoco.mj_id2name(model, obj, index)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(WBR_XML_PATH))


def test_mjcf_topology_and_no_external_assets(model: mujoco.MjModel) -> None:
  assert (model.nq, model.nv, model.nu, model.nbody, model.njnt) == (21, 20, 8, 16, 15)
  assert (model.neq, model.ntendon, model.ngeom) == (4, 2, 7)
  xml = WBR_XML_PATH.read_text()
  assert not re.search(r"(?:file=|mesh=|/home/|/Users/|[A-Za-z]:\\)", xml)
  assert all(
    model.geom_group[gid] <= 2
    for gid in range(model.ngeom)
    if (_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith("collision_")
  )
  with_floor = xml.replace(
    "<worldbody>", '<worldbody><geom name="terrain" type="plane" size="0 0 0.1"/>'
  )
  assert mujoco.MjModel.from_xml_string(with_floor).ngeom == 8


def test_motor_order_axes_limits_and_home(model: mujoco.MjModel) -> None:
  actuators = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
  assert tuple(actuators[:6]) == MOTOR_ACTUATOR_NAMES
  expected_axes = ((0, -1, 0), (0, -1, 0), (0, -1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0))
  for name, axis, limit in zip(ACTIVE_JOINT_NAMES, expected_axes, TORQUE_LIMITS, strict=True):
    jid = model.joint(name).id
    np.testing.assert_allclose(model.jnt_axis[jid], axis)
    np.testing.assert_allclose(model.jnt_actfrcrange[jid], (-limit, limit))
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
  np.testing.assert_allclose(model.actuator_forcerange[:6], [(-40, 40), (-40, 40), (-5.2, 5.2)] * 2)
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
  assert sequence_runner_cfg("plane").seed == 1


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


def test_onnx_contract_matches_torch(tmp_path: Path) -> None:
  ort = pytest.importorskip("onnxruntime")
  torch.manual_seed(7)
  policy = SequencePolicy().eval()
  output = tmp_path / "policy.onnx"
  export_onnx(policy, output)
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
    obs, *_ = env.step(torch.randn(4, 6) * 0.5)
    assert all(torch.isfinite(value).all() for value in obs.values())
  finally:
    env.close()
