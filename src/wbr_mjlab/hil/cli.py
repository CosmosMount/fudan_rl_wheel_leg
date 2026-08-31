"""MuJoCo hardware-in-the-loop simulation using STM32 USB policy inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..robot import POLICY_DT
from ..sim2sim import KeyboardControl, NativeRunner
from .client import HilClient, HilError, HilTransportError, UsbPolicy

EXPECTED_USB_VID = 0x0483
EXPECTED_USB_PID = 0x5710
EXPECTED_USB_PRODUCT = "WBR-H723-HIL-v1"


def _check_usb_identity(port: str) -> None:
  """Report the selected tty and reject a definitely different USB device."""
  try:
    from serial.tools import list_ports
  except ImportError:
    return

  target = Path(port).resolve()
  info = next(
    (candidate for candidate in list_ports.comports() if Path(candidate.device).resolve() == target),
    None,
  )
  if info is None:
    return
  usb_id = (
    f"{info.vid:04x}:{info.pid:04x}"
    if info.vid is not None and info.pid is not None
    else "unknown VID:PID"
  )
  product = info.product or info.description or "unknown product"
  print(f"USB serial {info.device} | {usb_id} | {product}")
  if info.vid is not None and info.pid is not None and (
    info.vid != EXPECTED_USB_VID or info.pid != EXPECTED_USB_PID
  ):
    raise HilTransportError(
      f"{info.device} is {usb_id} ({product}), but the WBR HIL USB Device firmware "
      f"must enumerate as {EXPECTED_USB_VID:04x}:{EXPECTED_USB_PID:04x}. "
      "Do not select the ST-LINK virtual COM port."
    )
  if info.product and info.product != EXPECTED_USB_PRODUCT:
    raise HilTransportError(
      f"{info.device} reports product {info.product!r}; the current HIL firmware must report "
      f"{EXPECTED_USB_PRODUCT!r}. Rebuild and flash /home/z/code/wbr-rl-deploy."
    )


def _uint32(text: str) -> int:
  value = int(text, 0)
  if not 0 < value <= 0xFFFFFFFF:
    raise argparse.ArgumentTypeError("session must be a non-zero uint32")
  return value


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", default="/dev/ttyACM0", help="USB CDC serial device")
  parser.add_argument("--baudrate", type=int, default=115_200)
  parser.add_argument("--timeout-ms", type=float, default=10.0, help="Whole request deadline")
  parser.add_argument(
    "--handshake-timeout-ms",
    type=float,
    default=500.0,
    help="Initial HELLO deadline; inference continues to use --timeout-ms",
  )
  parser.add_argument("--session", type=_uint32, help="Fixed session id (default: random)")
  parser.add_argument("--mode", choices=("plane", "jump"), default="plane")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=2000, help="Headless control steps at 100 Hz")
  parser.add_argument("--vx", type=float, default=0.0, help="Headless forward command, m/s")
  parser.add_argument("--yaw", type=float, default=0.0, help="Headless yaw command, rad/s")
  parser.add_argument("--height", type=float, help="Headless root-height command, m")
  parser.add_argument("--velocity", type=float, default=0.8, help="Keyboard speed, m/s")
  parser.add_argument("--yaw-rate", type=float, default=1.5, help="Keyboard yaw rate, rad/s")
  parser.add_argument("--output", type=Path, help="Headless trajectory .npz")
  parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pace headless control steps against the 100 Hz wall clock",
  )
  args = parser.parse_args(argv)
  height_range = (0.10, 0.20) if args.mode == "plane" else (0.12, 0.15)
  args.height = args.height if args.height is not None else sum(height_range) / 2
  numeric = (
    args.timeout_ms,
    args.handshake_timeout_ms,
    args.vx,
    args.yaw,
    args.height,
    args.velocity,
    args.yaw_rate,
  )
  max_vx = 2.0 if args.mode == "plane" else 2.1
  valid = (
    all(np.isfinite(value) for value in numeric)
    and args.timeout_ms > 0
    and args.handshake_timeout_ms > 0
    and args.baudrate > 0
    and args.steps > 0
    and abs(args.vx) <= max_vx
    and abs(args.yaw) <= 2.0
    and height_range[0] <= args.height <= height_range[1]
    and 0 < args.velocity <= 2.0
    and 0 < args.yaw_rate <= 2.0
  )
  if not valid:
    parser.error("Commands, timing and serial settings must be finite and within valid ranges")
  if args.output is not None and not args.headless:
    parser.error("--output is available with --headless")
  return args


def _headless(
  args,
  runner: NativeRunner,
  client: HilClient,
  *,
  clock=time.perf_counter,
  sleep=time.sleep,
) -> dict:
  command = np.array([args.vx, args.yaw, args.height], dtype=np.float32)
  names = (
    "time",
    "qpos",
    "qvel",
    "command",
    "obs",
    "history",
    "action",
    "torque",
    "round_trip_us",
    "inference_us",
  )
  trajectory = {name: [] for name in names}
  start = next_step = clock()
  min_height = float("inf")
  deadline_misses = 0
  for index in range(args.steps):
    if args.realtime:
      if index:
        next_step += POLICY_DT
      delay = next_step - clock()
      if delay > 0:
        sleep(delay)
    runner.step(command)
    now = clock()
    if args.realtime and now > next_step + POLICY_DT:
      deadline_misses += 1
      # Restart the wall-clock schedule after an overrun. Never emit a burst of
      # short control periods to catch up, and never skip a MuJoCo control step.
      next_step = now
    min_height = min(min_height, runner.data.qpos[runner.root_qadr + 2])
    inference = client.last_result
    if inference is None:
      raise RuntimeError("HIL client completed a step without inference timing")
    if args.output:
      values = (
        runner.data.time,
        runner.data.qpos,
        runner.data.qvel,
        command,
        runner.last_obs[0],
        runner.history.reshape(125),
        runner.action,
        runner.torque,
        inference.round_trip_us,
        inference.inference_us,
      )
      for name, value in zip(names, values, strict=True):
        trajectory[name].append(np.array(value, copy=True))
  elapsed = clock() - start
  result = {
    "backend": "stm32_usb_hil",
    "mode": args.mode,
    "port": args.port,
    "session": client.session,
    "steps": args.steps,
    "sim_seconds": runner.data.time,
    "wall_seconds": elapsed,
    "realtime_factor": runner.data.time / elapsed,
    "realtime_pacing": args.realtime,
    "deadline_misses": deadline_misses,
    "min_root_height": min_height,
    "final_root_pos": runner.data.qpos[runner.root_qadr : runner.root_qadr + 3].tolist(),
    "timing": client.timing_summary(),
    "finite": True,
  }
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **trajectory, metadata=json.dumps(result))
  return result


def run(args) -> dict:
  _check_usb_identity(args.port)
  with HilClient(
    port=args.port,
    baudrate=args.baudrate,
    timeout=args.timeout_ms / 1000.0,
    handshake_timeout=args.handshake_timeout_ms / 1000.0,
    session=args.session,
  ) as client:
    policies = {mode: UsbPolicy(client, mode) for mode in ("plane", "jump")}
    runner = NativeRunner(policies[args.mode], args.mode, policies=policies)
    hello = client.handshake()
    print(
      f"MuJoCo / STM32 USB HIL | session {client.session:#010x} | "
      f"capabilities {hello.capabilities:#x} | physics 1000 Hz / policy 100 Hz"
    )
    if args.headless:
      return _headless(args, runner, client)
    keyboard = KeyboardControl(
      args.mode,
      args.velocity,
      args.yaw_rate,
      available_modes=policies,
    )
    from ..sim2sim_viewer import run_viewer

    run_viewer(runner, keyboard, backend_name="STM32 USB HIL")
    return {
      "backend": "stm32_usb_hil",
      "port": args.port,
      "session": client.session,
      "sim_seconds": runner.data.time,
      "timing": client.timing_summary(),
    }


def main() -> None:
  args = parse_args()
  try:
    result = run(args)
  except (HilError, FloatingPointError) as exc:
    print(f"HIL stopped safely: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
