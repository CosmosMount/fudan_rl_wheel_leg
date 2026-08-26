"""External mjlab task package for the WBR wheel-legged robot."""

from mjlab.tasks.registry import register_mjlab_task

from .rl import SequenceRunner, sequence_runner_cfg
from .task import make_env_cfg

PLANE_TASK_ID = "Mjlab-Velocity-Flat-WBR"
JUMP_TASK_ID = "Mjlab-Jump-Flat-WBR"

register_mjlab_task(
  task_id=PLANE_TASK_ID,
  env_cfg=make_env_cfg("plane"),
  play_env_cfg=make_env_cfg("plane", play=True),
  rl_cfg=sequence_runner_cfg("plane"),
  runner_cls=SequenceRunner,
)

register_mjlab_task(
  task_id=JUMP_TASK_ID,
  env_cfg=make_env_cfg("jump"),
  play_env_cfg=make_env_cfg("jump", play=True),
  rl_cfg=sequence_runner_cfg("jump"),
  runner_cls=SequenceRunner,
)

__all__ = ["JUMP_TASK_ID", "PLANE_TASK_ID", "make_env_cfg"]
