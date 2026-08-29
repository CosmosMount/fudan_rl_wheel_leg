"""Measure WBR rollout throughput without running an optimizer or saving a policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from wbr_mjlab.task import make_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("plane", "jump"), default="plane")
  parser.add_argument("--num-envs", type=int, default=8192)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--steps", type=int, default=48)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--nconmax", type=int)
  parser.add_argument("--njmax", type=int)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--tolerance", type=float)
  parser.add_argument("--profile", action="store_true")
  parser.add_argument(
    "--timings", action="store_true", help="Synchronize each component for timing"
  )
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.num_envs < 1 or args.steps < 1 or args.warmup < 0:
    parser.error("num-envs and steps must be positive; warmup must be non-negative")
  if args.profile and args.timings:
    parser.error("use --profile and --timings in separate runs")
  torch.set_num_threads(1)
  torch.manual_seed(1)
  cfg = make_env_cfg(args.mode)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = 1
  if args.nconmax is not None:
    cfg.sim.nconmax = args.nconmax
  if args.njmax is not None:
    cfg.sim.njmax = args.njmax
  if args.iterations is not None:
    cfg.sim.mujoco.iterations = args.iterations
  if args.tolerance is not None:
    cfg.sim.mujoco.tolerance = args.tolerance
  env = ManagerBasedRlEnv(cfg, device=args.device)
  synchronize = (
    (lambda: torch.cuda.synchronize(args.device))
    if args.device.startswith("cuda")
    else (lambda: None)
  )
  try:
    with torch.inference_mode():
      env.reset(seed=1)
      actions = torch.randn(args.num_envs, 6, device=args.device) * 0.5
      for _ in range(args.warmup):
        env.step(actions)
      synchronize()
      start = time.perf_counter()
      for _ in range(args.steps):
        obs, reward, *_ = env.step(actions)
      synchronize()
      elapsed = time.perf_counter() - start
      result = {
        "mode": args.mode,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seconds": elapsed,
        "env_steps_per_second": args.num_envs * args.steps / elapsed,
        "ms_per_control_step": elapsed * 1000 / args.steps,
        "finite": bool(torch.isfinite(reward).all())
        and all(bool(torch.isfinite(x).all()) for x in obs.values()),
        "cuda_graph": env.sim.use_cuda_graph,
        "nconmax": cfg.sim.nconmax,
        "njmax": cfg.sim.njmax,
        "iterations": cfg.sim.mujoco.iterations,
        "tolerance": cfg.sim.mujoco.tolerance,
      }
      import warp as wp

      result["final_state_diagnostics"] = {
        "max_constraints": int(wp.to_torch(env.sim.wp_data.nefc).max()),
        "total_contacts": int(wp.to_torch(env.sim.wp_data.nacon).sum()),
        "max_solver_iterations": int(wp.to_torch(env.sim.wp_data.solver_niter).max()),
        "mean_solver_iterations": float(wp.to_torch(env.sim.wp_data.solver_niter).float().mean()),
      }
      if args.profile:
        from torch.profiler import ProfilerActivity, profile, record_function

        def instrument(obj, name, label):
          original = getattr(obj, name)

          def wrapped(*a, **kw):
            with record_function(label):
              return original(*a, **kw)

          setattr(obj, name, wrapped)

        for obj, name, label in (
          (env.action_manager, "apply_action", "wbr/action"),
          (env.scene, "write_data_to_sim", "wbr/write_controls"),
          (env.sim, "step", "wbr/physics"),
          (env.sim, "forward", "wbr/forward"),
          (env.reward_manager, "compute", "wbr/rewards"),
          (env.observation_manager, "compute", "wbr/observations"),
          (env.termination_manager, "compute", "wbr/terminations"),
        ):
          instrument(obj, name, label)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
          for _ in range(5):
            env.step(actions)
          synchronize()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
        result["profile"] = [
          {"name": e.key, "calls": e.count, "cpu_total_ms": e.cpu_time_total / 1000}
          for e in prof.key_averages()
          if e.key.startswith("wbr/")
        ]
      if args.timings:
        totals = {}

        def time_component(obj, name, label):
          original = getattr(obj, name)

          def wrapped(*a, **kw):
            synchronize()
            start = time.perf_counter()
            value = original(*a, **kw)
            synchronize()
            totals[label] = totals.get(label, 0.0) + time.perf_counter() - start
            return value

          setattr(obj, name, wrapped)

        for obj, name, label in (
          (env.action_manager, "apply_action", "action"),
          (env.scene, "write_data_to_sim", "write_controls"),
          (env.sim, "step", "physics"),
          (env.sim, "forward", "forward"),
          (env.sim, "sense", "sensors"),
          (env.reward_manager, "compute", "rewards"),
          (env.observation_manager, "compute", "observations"),
          (env.termination_manager, "compute", "terminations"),
        ):
          time_component(obj, name, label)
        for _ in range(5):
          env.step(actions)
        result["synchronized_ms_per_control_step"] = {
          key: value * 1000 / 5 for key, value in totals.items()
        }
      print(json.dumps(result, indent=2))
      if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  main()
