from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.buffers import CircularBuffer

from wbr_mjlab.mdp import hybrid_torque, policy_vector, quat_rotate_inverse
from wbr_mjlab.robot import HOME_ACTIVE_JOINT_POS, TORQUE_LIMITS
from wbr_mjlab.sim2sim import (
  KeyboardControl,
  NativeRunner,
  grounded_wheel_count,
  motor_torque,
  observation_frame,
  parse_args,
  wheels_grounded,
)
from wbr_mjlab.task import make_env_cfg


def test_native_scene_and_initial_observations_match_mjlab():
  from mjlab.envs import ManagerBasedRlEnv

  cfg = make_env_cfg("plane", play=True)
  cfg.seed = 1
  cfg.events.pop("reset_velocity")
  env = ManagerBasedRlEnv(cfg, device="cpu")
  runner = NativeRunner(lambda obs, hist: np.zeros(6), "plane")
  try:
    obs, _ = env.reset(seed=1)
    command = env.command_manager.get_command("motion")[0].numpy()
    native_obs, native_history = runner.observe(command)
    np.testing.assert_array_equal(native_obs, obs["policy"].numpy())
    np.testing.assert_array_equal(native_history, obs["history"].numpy())
    reference = env.sim.mj_model
    for kind, count, fields in (
      (
        "geom",
        runner.model.ngeom,
        ("contype", "conaffinity", "friction", "solref", "solimp", "condim"),
      ),
      ("body", runner.model.nbody, ("mass", "inertia", "ipos", "iquat")),
      ("actuator", runner.model.nu, ("gainprm", "biasprm", "ctrlrange", "forcerange", "gear")),
    ):
      for i in range(count):
        item = getattr(runner.model, kind)(i)
        if not item.name or item.name == "world":
          continue
        other = getattr(reference, kind)(
          "terrain" if item.name == "terrain" else "robot/" + item.name
        )
        for field in fields:
          np.testing.assert_allclose(getattr(item, field), getattr(other, field), atol=1e-8)
    for field in ("timestep", "integrator", "solver", "iterations", "tolerance", "cone", "gravity"):
      np.testing.assert_array_equal(getattr(runner.model.opt, field), getattr(reference.opt, field))
    terrain_id = runner.model.geom("terrain").id
    material_id = runner.model.geom_matid[terrain_id]
    assert material_id >= 0
    assert runner.model.mat_texid[material_id, mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0
  finally:
    env.close()


@pytest.mark.parametrize("mode", ("plane", "jump"))
def test_numpy_observation_and_pd_match_training(mode):
  rng = np.random.default_rng(24)
  default = torch.tensor(HOME_ACTIVE_JOINT_POS)[None]
  for _ in range(50):
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)
    ang, cmd = rng.normal(size=(2, 3)).astype(np.float32)
    q, qd, act = rng.normal(size=(3, 6)).astype(np.float32)
    qd *= 40
    gravity = quat_rotate_inverse(torch.tensor(quat)[None], torch.tensor([[0.0, 0.0, -1.0]]))
    expected = policy_vector(
      torch.tensor(ang)[None],
      gravity,
      torch.tensor(cmd)[None],
      torch.tensor(q)[None],
      default,
      torch.tensor(qd)[None],
      torch.tensor(act)[None],
      mode,
    )
    np.testing.assert_allclose(
      observation_frame(ang, quat, cmd, q, qd, act, mode), expected[0].numpy(), atol=2e-6
    )
    scale = torch.ones(1, 6)
    torque = hybrid_torque(
      torch.tensor(act)[None],
      torch.tensor(q)[None],
      torch.tensor(qd)[None],
      default,
      scale,
      scale,
      scale,
      mode,
    )
    np.testing.assert_allclose(motor_torque(act, q, qd, mode), torque[0].numpy(), atol=2e-6)


def test_native_history_and_reset_match_mjlab_buffer():
  runner = NativeRunner(lambda obs, hist: np.zeros(6), "plane")
  buffer = CircularBuffer(5, 1, "cpu")
  for i in range(9):
    # An intentionally large input checks that only the current frame is clipped.
    runner.data.qvel[runner.root_vadr + 3] = i * 800
    cmd = np.array([i * 0.1, 0, 0.15], dtype=np.float32)
    frame = observation_frame(
      runner.data.qvel[3:6], runner.data.qpos[3:7], cmd, runner.q, runner.qd, runner.action, "plane"
    )
    buffer.append(torch.tensor(frame)[None])
    obs, history = runner.observe(cmd)
    np.testing.assert_array_equal(history, buffer.buffer.reshape(1, 125).numpy())
    np.testing.assert_array_equal(obs, np.clip(frame[None], -100, 100))
  runner.reset()
  assert runner.history is None and not runner.action.any() and not runner.qd.any()
  obs, history = runner.observe(np.array([0, 0, 0.15]))
  np.testing.assert_array_equal(history, np.tile(obs, (1, 5)))


@pytest.mark.parametrize("mode", ("plane", "jump"))
def test_native_substep_timing_and_motor_mapping(mode, monkeypatch):
  action = np.array([0.1, -0.15, 0.05, -0.1, 0.15, -0.05], dtype=np.float32)
  runner = NativeRunner(lambda obs, hist: action, mode)
  original_step = mujoco.mj_step
  previous_q = runner.q.copy()
  expected_qd = np.zeros(6, dtype=np.float32)
  calls = 0

  def checked_step(model, data):
    nonlocal calls, previous_q, expected_qd
    np.testing.assert_array_equal(runner.qd, expected_qd)
    np.testing.assert_allclose(
      data.ctrl[runner.motor_ids], motor_torque(action, previous_q, expected_qd, mode), atol=1e-6
    )
    assert data.ctrl[model.actuator("left_gas_spring_actuator").id] == 0
    assert data.ctrl[model.actuator("right_gas_spring_actuator").id] == 0
    original_step(model, data)
    q = data.qpos[runner.qadr].astype(np.float32)
    expected_qd = ((q - previous_q + np.pi) % (2 * np.pi) - np.pi) / 0.001
    previous_q = q
    calls += 1

  monkeypatch.setattr(mujoco, "mj_step", checked_step)
  for _ in range(5):
    runner.step(np.array([0, 0, 0.15]))
  assert calls == 50 and runner.data.time == pytest.approx(0.05)
  np.testing.assert_allclose(runner.history[-1, -6:], action)
  assert (np.abs(runner.torque) <= TORQUE_LIMITS).all()
  monkeypatch.setattr(mujoco, "mj_step", original_step)
  runner.step(np.array([0, 0, 0.15]), enabled=False)
  assert not runner.data.ctrl.any() and runner.history is None
  assert runner.data.time == pytest.approx(0.06)  # Disabled motors do not pause physics.


def test_keyboard_edges_release_focus_and_command_limits():
  keyboard = KeyboardControl("plane")
  assert keyboard.paused and not keyboard.enabled
  keyboard.key(257, 1)
  keyboard.key(257, 2)  # Autorepeat must not toggle the enable latch.
  assert keyboard.enabled and not keyboard.paused
  keyboard.key(ord("W"), 1)
  keyboard.key(ord("A"), 1)
  np.testing.assert_allclose(keyboard.command(), [0.8, 1.5, 0.15])
  keyboard.key(ord("S"), 1)
  keyboard.key(ord("D"), 1)
  np.testing.assert_allclose(keyboard.command(), [0, 0, 0.15])
  keyboard.key(ord("Q"), 1)
  assert keyboard.command()[2] == pytest.approx(0.1)
  keyboard.key(ord("F"), 1)
  assert keyboard.command()[2] == pytest.approx(0.2)
  keyboard.key(344, 1)
  np.testing.assert_allclose(keyboard.command()[:2], [0, 1.5])
  keyboard.key(344, 0)
  keyboard.lose_focus()
  np.testing.assert_allclose(keyboard.command()[:2], [0, 0])
  assert keyboard.enabled  # Keep the balancing policy alive on focus loss.
  keyboard.key(32, 1)
  assert "No jump policy loaded" in keyboard.notice
  assert keyboard.pending_mode is None and keyboard.mode == "plane"
  keyboard.key(ord("G"), 1)
  assert "No stair policy" in keyboard.notice
  keyboard.key(257, 1)
  assert not keyboard.enabled
  keyboard.key(259, 1)
  assert keyboard.reset_requested
  keyboard.reset()
  assert not keyboard.held and keyboard.paused and not keyboard.reset_requested
  jump = KeyboardControl("jump")
  jump.key(ord("F"), 1)
  assert jump.command()[2] == pytest.approx(0.15)


def test_wheel_ground_contact_requires_both_wheels_on_terrain():
  wheel_ids = frozenset((11, 12))
  left = SimpleNamespace(geom1=3, geom2=11)
  right_reversed = SimpleNamespace(geom1=12, geom2=3)
  unrelated = SimpleNamespace(geom1=3, geom2=99)
  assert grounded_wheel_count((), 3, wheel_ids) == 0
  assert grounded_wheel_count((left, unrelated), 3, wheel_ids) == 1
  assert not wheels_grounded((left, unrelated), 3, wheel_ids)
  assert wheels_grounded((left, right_reversed, unrelated), 3, wheel_ids)


def test_space_runs_one_jump_then_returns_to_plane_after_stable_landing():
  class Runner:
    def __init__(self):
      self.mode = "plane"
      self.contact_count = 2

    def switch_policy(self, mode):
      changed = mode != self.mode
      self.mode = mode
      return changed

    def wheels_grounded(self):
      return self.contact_count == 2

    def grounded_wheel_count(self):
      return self.contact_count

  runner = Runner()
  keyboard = KeyboardControl("plane", available_modes=("plane", "jump"))
  keyboard.key(257, 1)
  keyboard.key(32, 1)
  keyboard.key(32, 0)
  assert keyboard.pending_mode == "jump" and keyboard.pending_jump_once
  keyboard.apply_policy_request(runner)
  assert keyboard.mode == runner.mode == "jump"
  assert keyboard.jump_once_phase == "takeoff" and keyboard.jump_once_ground_seen

  keyboard.key(32, 1)
  keyboard.key(32, 0)
  assert keyboard.pending_mode is None
  assert keyboard.jump_once_phase == "takeoff"

  for contact_count in (0, 1, 0, 0):
    runner.contact_count = contact_count
    keyboard.update_jump_once(runner)
  assert keyboard.jump_once_phase == "landing"

  for _ in range(2):
    runner.contact_count = 2
    keyboard.update_jump_once(runner)
    assert keyboard.pending_mode is None
  keyboard.update_jump_once(runner)
  assert keyboard.pending_mode == "plane" and keyboard.jump_once_phase == "returning"
  keyboard.apply_policy_request(runner)
  assert keyboard.mode == runner.mode == "plane"
  assert keyboard.jump_once_phase is None


@pytest.mark.parametrize("initial,target", (("plane", "jump"), ("jump", "plane")))
def test_policy_switch_keeps_physics_and_rebuilds_history(initial, target, monkeypatch):
  captured = {}
  first_action = np.array([0.1, -0.1, 0.02, -0.1, 0.1, -0.02], dtype=np.float32)
  next_action = first_action * 2

  def next_policy(obs, history):
    captured["obs"], captured["history"] = obs.copy(), history.copy()
    return next_action

  runner = NativeRunner(lambda obs, hist: first_action, initial, policies={target: next_policy})
  keyboard = KeyboardControl(initial, available_modes=runner.policies)
  keyboard.key(257, 1)
  keyboard.key(ord("W"), 1)
  keyboard.key(ord("F"), 1)
  runner.step(keyboard.command())
  old_qpos, old_qvel = runner.data.qpos.copy(), runner.data.qvel.copy()
  old_q, old_qd, old_ctrl = runner.q.copy(), runner.qd.copy(), runner.data.ctrl.copy()
  old_time = runner.data.time
  keyboard.key(32, 1)
  keyboard.key(32, 2)
  assert keyboard.pending_mode == target
  keyboard.apply_policy_request(runner)
  assert runner.mode == keyboard.mode == target and keyboard.enabled and not keyboard.paused
  assert runner.policy is next_policy and runner.history is None
  np.testing.assert_array_equal(runner.action, first_action)
  np.testing.assert_array_equal(runner.data.qpos, old_qpos)
  np.testing.assert_array_equal(runner.data.qvel, old_qvel)
  np.testing.assert_array_equal(runner.qd, old_qd)
  np.testing.assert_array_equal(runner.data.ctrl, old_ctrl)
  assert runner.data.time == old_time
  command = keyboard.command()
  assert command[0] == pytest.approx(0.8)
  assert command[2] == pytest.approx(0.135 if target == "jump" else 0.15)
  keyboard.key(32, 2)
  assert keyboard.pending_mode is None  # Holding Space does not switch back.
  original_step = mujoco.mj_step

  def checked_step(model, data):
    if data.time == old_time:
      np.testing.assert_allclose(
        data.ctrl[runner.motor_ids], motor_torque(next_action, old_q, old_qd, target), atol=1e-6
      )
    original_step(model, data)

  monkeypatch.setattr(mujoco, "mj_step", checked_step)
  runner.step(command)
  expected = observation_frame(
    old_qvel[3:6], old_qpos[3:7], command, old_q, old_qd, first_action, target
  )
  np.testing.assert_allclose(captured["obs"], expected[None])
  np.testing.assert_allclose(captured["history"], np.tile(expected, (1, 5)))
  history = runner.history
  assert not runner.switch_policy(target) and runner.history is history
  assert runner.data.time == pytest.approx(old_time + 0.01)
  keyboard.key(ord("1") if initial == "plane" else ord("2"), 1)
  keyboard.apply_policy_request(runner)
  assert keyboard.mode == runner.mode == initial
  keyboard.reset()
  assert keyboard.mode == initial and keyboard.pending_mode is None


def test_missing_policy_does_not_modify_runner():
  runner = NativeRunner(lambda obs, hist: np.zeros(6), "jump")
  runner.step(np.array([0, 0, 0.135]))
  history = runner.history
  with pytest.raises(ValueError, match="No plane policy loaded"):
    runner.switch_policy("plane")
  assert runner.mode == "jump" and runner.history is history


def test_single_and_dual_policy_cli():
  single = parse_args(["--mode", "jump", "--onnx", "jump.onnx"])
  assert single.policy_paths == {"jump": single.onnx}
  assert single.height == pytest.approx(0.135)
  dual = parse_args(["--plane-onnx", "plane.onnx", "--jump-onnx", "jump.onnx"])
  assert dual.mode == "plane" and set(dual.policy_paths) == {"plane", "jump"}
  mixed = parse_args(["--mode", "jump", "--onnx", "jump.onnx", "--plane-onnx", "plane.onnx"])
  assert mixed.policy_paths == dual.policy_paths
  high_command = parse_args(
    [
      "--onnx",
      "plane.onnx",
      "--vx",
      "3",
      "--yaw",
      "8",
      "--velocity",
      "3",
      "--yaw-rate",
      "8",
    ]
  )
  assert (high_command.vx, high_command.yaw) == (3.0, 8.0)
  assert (high_command.velocity, high_command.yaw_rate) == (3.0, 8.0)
  for args in (
    [],
    ["--jump-onnx", "jump.onnx"],
    ["--onnx", "a.onnx", "--plane-onnx", "b.onnx"],
    ["--mode", "jump", "--onnx", "jump.onnx", "--plane-onnx", "plane.onnx", "--velocity", "3.01"],
    ["--onnx", "plane.onnx", "--vx", "3.01"],
    ["--onnx", "plane.onnx", "--yaw", "8.01"],
  ):
    with pytest.raises(SystemExit) as exc:
      parse_args(args)
    assert exc.value.code == 2


def test_registered_mouse_callbacks_use_real_mujoco_camera_api(monkeypatch):
  import glfw

  from wbr_mjlab.sim2sim_viewer import install_callbacks

  callbacks = {}
  for name in ("key", "window_focus", "cursor_pos", "scroll"):
    monkeypatch.setattr(
      glfw, f"set_{name}_callback", lambda win, cb, name=name: callbacks.update({name: cb})
    )
  pressed = False
  monkeypatch.setattr(glfw, "get_cursor_pos", lambda win: (0, 0))
  monkeypatch.setattr(glfw, "get_window_size", lambda win: (1280, 800))
  monkeypatch.setattr(
    glfw, "get_mouse_button", lambda win, button: glfw.PRESS if pressed else glfw.RELEASE
  )
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
  )
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(camera)
  camera.distance, camera.azimuth, camera.elevation = 2.5, 90, -15
  keyboard = KeyboardControl("plane")
  install_callbacks(None, model, camera, keyboard)
  callbacks["cursor_pos"](None, 50, 30)
  assert camera.azimuth == 90 and camera.elevation == -15
  pressed = True
  callbacks["cursor_pos"](None, 100, 60)
  assert camera.azimuth != 90 and camera.elevation != -15
  callbacks["scroll"](None, 0, 1)
  assert 0 < camera.distance < 2.5
  zoomed_distance = camera.distance
  callbacks["scroll"](None, 0, -1)
  assert camera.distance > zoomed_distance
  callbacks["key"](None, glfw.KEY_ENTER, 0, glfw.PRESS, 0)
  callbacks["key"](None, glfw.KEY_W, 0, glfw.PRESS, 0)
  assert keyboard.command()[0] > 0
  callbacks["window_focus"](None, False)
  assert keyboard.command()[0] == 0
