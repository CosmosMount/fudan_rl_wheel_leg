"""External mjlab task package for the WBR wheel-legged robot."""

from mjlab.tasks.registry import register_mjlab_task

from .rl import SequenceRunner, sequence_runner_cfg
from .task import make_env_cfg

PLANE_TASK_ID = "Mjlab-Velocity-Flat-WBR"
JUMP_TASK_ID = "Mjlab-Jump-Flat-WBR"

for task_id, mode in ((PLANE_TASK_ID, "plane"), (JUMP_TASK_ID, "jump")):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_env_cfg(mode),
    play_env_cfg=make_env_cfg(mode, play=True),
    rl_cfg=sequence_runner_cfg(mode),
    runner_cls=SequenceRunner,
  )

__all__ = ["JUMP_TASK_ID", "PLANE_TASK_ID", "make_env_cfg"]
