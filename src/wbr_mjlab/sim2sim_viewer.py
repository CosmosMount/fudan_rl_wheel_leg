"""Small GLFW viewer with real key release events and a fixed simulation clock."""

from __future__ import annotations

import time

import glfw
import mujoco
import numpy as np

from .robot import POLICY_DT


def _next_policy_deadline(current: float, finished: float) -> float:
  scheduled = current + POLICY_DT
  return finished + POLICY_DT if finished > scheduled else scheduled


def install_callbacks(window, model, camera, keyboard) -> None:
  """Keep callbacks testable without a GL context; use the installed MuJoCo API."""
  mouse = np.array(glfw.get_cursor_pos(window))

  def cursor_callback(win, x, y):
    nonlocal mouse
    delta = np.array([x, y]) - mouse
    mouse = np.array([x, y])
    if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
      height = max(glfw.get_window_size(win)[1], 1)
      mujoco.mjv_moveCamera(
        model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, delta[0] / height, delta[1] / height, camera
      )

  def scroll_callback(_win, _x, y):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y, camera)

  glfw.set_key_callback(window, lambda _win, key, _scan, action, _mods: keyboard.key(key, action))
  glfw.set_window_focus_callback(
    window, lambda _win, focus: None if focus else keyboard.lose_focus()
  )
  glfw.set_cursor_pos_callback(window, cursor_callback)
  glfw.set_scroll_callback(window, scroll_callback)


def run_viewer(runner, keyboard, *, backend_name: str = "ONNX") -> None:
  if not glfw.init():
    raise RuntimeError("GLFW could not open a display; use --headless over SSH")
  window = glfw.create_window(1280, 800, f"WBR | MuJoCo + {backend_name}", None, None)
  if not window:
    glfw.terminate()
    raise RuntimeError("Could not create MuJoCo OpenGL window")
  context = None
  try:
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    model, data = runner.model, runner.data
    scene = mujoco.MjvScene(model, maxgeom=2000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 2.5, 90, -15
    camera.lookat[:] = data.qpos[runner.root_qadr : runner.root_qadr + 3]
    install_callbacks(window, model, camera, keyboard)
    next_step = next_frame = time.perf_counter()
    print(
      "Enter enable | WASD move | Q/E/F height | Space switch policy | 1 plane / 2 jump | "
      "Shift spin | X stop | P pause | Backspace reset | Esc quit"
    )
    while not glfw.window_should_close(window) and not keyboard.quit_requested:
      glfw.poll_events()
      if keyboard.reset_requested:
        runner.reset()
        keyboard.reset()
      keyboard.apply_policy_request(runner)
      now = time.perf_counter()
      if keyboard.paused:
        next_step = now
      elif now >= next_step:
        runner.step(keyboard.command(), enabled=keyboard.enabled)
        finished = time.perf_counter()
        # A slow inference runs simulation slower; do not skip controller updates or
        # issue back-to-back USB requests to catch up with an expired wall-clock target.
        next_step = _next_policy_deadline(next_step, finished)
        now = finished
      if now >= next_frame:
        width, height = glfw.get_framebuffer_size(window)
        if width > 0 and height > 0:
          viewport = mujoco.MjrRect(0, 0, width, height)
          camera.lookat[:] = data.qpos[runner.root_qadr : runner.root_qadr + 3]
          mujoco.mjv_updateScene(
            model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
          )
          mujoco.mjr_render(viewport, scene, context)
          command = keyboard.command()
          state = (
            "PAUSED"
            if keyboard.paused
            else (f"{backend_name} ENABLED" if keyboard.enabled else "MOTORS OFF")
          )
          status = (
            f"{runner.mode} | {state} | t={data.time:.2f}s\n"
            f"Loaded policies: {', '.join(runner.policies)}\n"
            f"vx {command[0]:+.2f} m/s | yaw {command[1]:+.2f} rad/s | height {command[2]:.3f} m\n"
            f"root z {data.qpos[runner.root_qadr + 2]:.3f} m\n{keyboard.notice}"
          )
          help_text = (
            "Enter enable/disable | W/S forward/back | A/D turn\n"
            "Q/E/F low/mid/high | Shift spin | X stop\n"
            "P pause | Backspace reset | Esc exit | Mouse drag/scroll camera\n"
            "Space switch plane/jump | 1 plane / 2 jump | G: no stair policy"
          )
          mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            status,
            "",
            context,
          )
          mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            help_text,
            "",
            context,
          )
          glfw.swap_buffers(window)
        next_frame = now + 1 / 60
      deadline = next_frame if keyboard.paused else min(next_frame, next_step)
      time.sleep(max(0.0, min(0.002, deadline - time.perf_counter())))
  finally:
    if context is not None:
      context.free()
    glfw.destroy_window(window)
    glfw.terminate()
