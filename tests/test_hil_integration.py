from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from wbr_mjlab.hil.cli import _headless, parse_args
from wbr_mjlab.hil.client import HilTimeoutError
from wbr_mjlab.hil.protocol import ACTION_DIM, OBS_DIM
from wbr_mjlab.sim2sim import NativeRunner
from wbr_mjlab.sim2sim_viewer import _next_policy_deadline, configure_gl_gpu


def test_hil_cli_accepts_high_speed_commands_and_rejects_overflow():
  args = parse_args(["--vx", "3", "--yaw", "8", "--velocity", "3", "--yaw-rate", "8"])
  assert (args.vx, args.yaw, args.velocity, args.yaw_rate) == (3.0, 8.0, 3.0, 8.0)
  assert args.gpu == "nvidia"
  assert parse_args(["--gpu", "system"]).gpu == "system"
  for extra in (("--vx", "3.01"), ("--yaw", "8.01")):
    with pytest.raises(SystemExit) as exc:
      parse_args(list(extra))
    assert exc.value.code == 2


def test_viewer_gpu_selection_defaults_to_nvidia_and_can_restore_system(monkeypatch):
  monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
  monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
  configure_gl_gpu("nvidia")
  assert os.environ["__NV_PRIME_RENDER_OFFLOAD"] == "1"
  assert os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"
  configure_gl_gpu("system")
  assert "__NV_PRIME_RENDER_OFFLOAD" not in os.environ
  assert "__GLX_VENDOR_LIBRARY_NAME" not in os.environ


def test_native_runner_rolls_back_history_and_disarms_action_on_policy_timeout():
  action = np.array([0.1, -0.1, 0.03, -0.1, 0.1, -0.03], dtype=np.float32)
  calls = 0

  def policy(_obs, _history):
    nonlocal calls
    calls += 1
    if calls == 1:
      return action
    raise HilTimeoutError("injected timeout")

  runner = NativeRunner(policy, "plane")
  command = np.array([0.0, 0.0, 0.15], dtype=np.float32)
  runner.step(command)
  before_time = runner.data.time
  before_history = runner.history.copy()
  before_obs = runner.last_obs.copy()
  assert np.any(runner.data.ctrl[runner.motor_ids])
  with pytest.raises(HilTimeoutError):
    runner.step(command)
  assert runner.data.time == before_time
  np.testing.assert_array_equal(runner.history, before_history)
  np.testing.assert_array_equal(runner.last_obs, before_obs)
  assert not runner.action.any() and not runner.torque.any()
  assert not runner.data.ctrl.any()


def test_headless_realtime_pacing_restarts_schedule_after_overrun_without_burst():
  class Clock:
    def __init__(self):
      self.now = 0.0
      self.sleeps = []

    def __call__(self):
      return self.now

    def sleep(self, duration):
      self.sleeps.append(duration)
      self.now += duration

  class Runner:
    def __init__(self, clock):
      self.clock = clock
      self.durations = iter((0.002, 0.015, 0.002, 0.002))
      self.starts = []
      self.root_qadr = 0
      self.data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6), time=0.0)
      self.last_obs = np.zeros((1, OBS_DIM), dtype=np.float32)
      self.history = np.zeros((5, OBS_DIM), dtype=np.float32)
      self.action = np.zeros(ACTION_DIM, dtype=np.float32)
      self.torque = np.zeros(ACTION_DIM, dtype=np.float32)

    def step(self, _command):
      self.starts.append(self.clock.now)
      self.clock.now += next(self.durations)
      self.data.time += 0.01
      client.last_result = SimpleNamespace(round_trip_us=500.0, inference_us=400)

  clock = Clock()
  client = SimpleNamespace(
    session=7,
    last_result=None,
    timing_summary=lambda: {"round_trip_us": None, "inference_us": None},
  )
  args = SimpleNamespace(
    vx=0.0,
    yaw=0.0,
    height=0.15,
    steps=4,
    realtime=True,
    output=None,
    mode="plane",
    port="memory",
  )
  runner = Runner(clock)
  result = _headless(args, runner, client, clock=clock, sleep=clock.sleep)
  assert result["deadline_misses"] == 1
  np.testing.assert_allclose(runner.starts, [0.0, 0.01, 0.035, 0.045], atol=1e-12)
  np.testing.assert_allclose(clock.sleeps, [0.008, 0.01, 0.008], atol=1e-12)


def test_viewer_deadline_waits_a_full_period_after_overrun():
  assert _next_policy_deadline(1.0, 1.002) == pytest.approx(1.01)
  assert _next_policy_deadline(1.0, 1.015) == pytest.approx(1.025)
