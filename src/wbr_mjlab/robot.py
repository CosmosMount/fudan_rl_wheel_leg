"""WBR model definition and controller-facing conventions."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = Path("assets/rm26_pnx_wbr_mjcf/mjmodel.xml")
_SOURCE_XML_PATH = PROJECT_ROOT / _MODEL_PATH
_PACKAGED_XML_PATH = Path(__file__).resolve().parent / _MODEL_PATH
WBR_XML_PATH = _SOURCE_XML_PATH if _SOURCE_XML_PATH.is_file() else _PACKAGED_XML_PATH

ACTIVE_JOINT_NAMES = (
  "ljoint1",
  "ljoint4",
  "lwheel",
  "rjoint1",
  "rjoint4",
  "rwheel",
)
LEG_IDS = (0, 1, 3, 4)
POLICY_DT = 0.01
MAX_FORWARD_COMMAND = 3.0  # m/s
MAX_YAW_COMMAND = 8.0  # rad/s
MOTOR_ACTUATOR_NAMES = tuple(f"{name}_actuator" for name in ACTIVE_JOINT_NAMES)
GAS_SPRING_NAMES = ("left_gas_spring", "right_gas_spring")
GAS_SPRING_ACTUATOR_NAMES = tuple(f"{name}_actuator" for name in GAS_SPRING_NAMES)
WHEEL_GEOM_NAMES = ("collision_lwheel", "collision_rwheel")
_MESH_GEOM_NAMES = {
  "base_link": "collision_base",
  "lwlink": "collision_lwheel",
  "rwlink": "collision_rwheel",
  **{
    name: f"collision_{name}"
    for side in ("l", "r")
    for name in (
      f"{side}link1",
      f"{side}link1_child1",
      f"{side}link4",
      f"{side}link4_child1",
      f"{side}link4_child2",
      f"{side}link4_child3",
    )
  },
}
ROBOT_COLLISION_GEOM_NAMES = tuple(_MESH_GEOM_NAMES.values())
PENALIZED_COLLISION_GEOM_NAMES = tuple(
  name for name in ROBOT_COLLISION_GEOM_NAMES if name not in WHEEL_GEOM_NAMES
)


def _home_from_xml() -> tuple[
  tuple[float, float, float],
  tuple[float, float, float, float],
  dict[str, float],
]:
  """Read the controller and reset reference directly from the MJCF home key."""
  spec = mujoco.MjSpec.from_file(str(WBR_XML_PATH))
  try:
    qpos = tuple(float(value) for value in spec.key("home").qpos)
  except KeyError as exc:
    raise ValueError(f"Robot MJCF must define <key name='home'>: {WBR_XML_PATH}") from exc

  offset = 0
  root_pos: tuple[float, float, float] | None = None
  root_rot: tuple[float, float, float, float] | None = None
  joint_pos: dict[str, float] = {}
  widths = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
  }
  for joint in spec.joints:
    width = widths[joint.type]
    values = qpos[offset : offset + width]
    if len(values) != width:
      raise ValueError(f"Robot MJCF home qpos is too short: {WBR_XML_PATH}")
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
      if root_pos is not None:
        raise ValueError("Robot MJCF home contains more than one free joint")
      root_pos = values[:3]
      root_rot = values[3:7]
    elif width == 1:
      if not joint.name:
        raise ValueError("Every scalar robot joint must be named")
      joint_pos[joint.name] = values[0]
    else:
      raise ValueError(f"Unsupported non-scalar home joint: {joint.name or '<unnamed>'}")
    offset += width
  if offset != len(qpos):
    raise ValueError(f"Robot MJCF home qpos has {len(qpos) - offset} trailing values")
  if root_pos is None or root_rot is None:
    raise ValueError("Robot MJCF home must include a free-joint root pose")
  missing = set(ACTIVE_JOINT_NAMES) - joint_pos.keys()
  if missing:
    raise ValueError(f"Robot MJCF home is missing active joints: {sorted(missing)}")
  return root_pos, root_rot, joint_pos


HOME_ROOT_POS, HOME_ROOT_ROT, HOME_JOINT_POS = _home_from_xml()
HOME_ACTIVE_JOINT_POS = tuple(HOME_JOINT_POS[name] for name in ACTIVE_JOINT_NAMES)
MOTOR_ZERO_RAD = (-0.06, -0.20, 0.0, 0.06, 0.20, 0.0)
TORQUE_LIMITS = (20.0, 20.0, 5.2, 20.0, 20.0, 5.2)
LEG_LINK_1 = 0.220
LEG_LINK_2 = 0.260
WHEEL_RADIUS = 0.060
IMU_OFFSET = (0.0, 0.0, 0.0)


def load_wbr_spec() -> mujoco.MjSpec:
  """Load the mesh model while preserving its XML-defined home keyframe."""
  spec = mujoco.MjSpec.from_file(str(WBR_XML_PATH))
  for geom in spec.geoms:
    geom.name = _MESH_GEOM_NAMES[geom.meshname]
    # Retain terrain-only collision filtering from the task. Upstream CAD convex
    # hulls overlap at closed-chain pivots (up to 17 mm at the reset pose).
    geom.contype = 2
    geom.conaffinity = 1
  return spec


def _load_wbr_entity_spec() -> mujoco.MjSpec:
  spec = load_wbr_spec()
  # EntityCfg writes an equivalent float32-safe init_state keyframe from the
  # values parsed from the source XML home key. Remove the source key only from
  # the composed runtime spec to avoid duplicate scene keyframes.
  for key in tuple(spec.keys):
    spec.delete(key)
  return spec


def get_wbr_robot_cfg() -> EntityCfg:
  """Create the mjlab entity config while retaining all XML actuators."""
  return EntityCfg(
    spec_fn=_load_wbr_entity_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=HOME_ROOT_POS,
      rot=HOME_ROOT_ROT,
      # Retain every passive closed-chain coordinate from the XML home pose.
      # mjlab's explicit map also keeps the MuJoCo Warp state float32-safe.
      joint_pos=HOME_JOINT_POS,
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=ACTIVE_JOINT_NAMES,
          transmission_type=TransmissionType.JOINT,
          command_field="effort",
        ),
        XmlActuatorCfg(
          target_names_expr=GAS_SPRING_NAMES,
          transmission_type=TransmissionType.TENDON,
          command_field="effort",
        ),
      ),
      soft_joint_pos_limit_factor=0.97,
    ),
  )
