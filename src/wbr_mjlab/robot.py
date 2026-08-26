"""WBR model definition and controller-facing conventions."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_XML_PATH = PROJECT_ROOT / "assets" / "wbr.xml"
_PACKAGED_XML_PATH = Path(__file__).resolve().parent / "assets" / "wbr.xml"
WBR_XML_PATH = _SOURCE_XML_PATH if _SOURCE_XML_PATH.is_file() else _PACKAGED_XML_PATH

ACTIVE_JOINT_NAMES = (
  "ljoint1",
  "ljoint4",
  "lwheel",
  "rjoint1",
  "rjoint4",
  "rwheel",
)
MOTOR_ACTUATOR_NAMES = tuple(f"{name}_actuator" for name in ACTIVE_JOINT_NAMES)
GAS_SPRING_NAMES = ("left_gas_spring", "right_gas_spring")
GAS_SPRING_ACTUATOR_NAMES = tuple(f"{name}_actuator" for name in GAS_SPRING_NAMES)
WHEEL_GEOM_NAMES = ("collision_lwheel", "collision_rwheel")
ROBOT_COLLISION_GEOM_NAMES = (
  "collision_lwheel",
  "collision_llink1",
  "collision_llink4",
  "collision_rwheel",
  "collision_rlink1",
  "collision_rlink4",
  "collision_base",
)
PENALIZED_COLLISION_GEOM_NAMES = (
  "collision_llink1",
  "collision_llink4",
  "collision_rlink1",
  "collision_rlink4",
  "collision_base",
)

HOME_ACTIVE_JOINT_POS = (0.312, -0.568, 0.0, -0.312, 0.568, 0.0)
HOME_JOINT_POS = {
  "ljoint1": 0.312,
  "llink1child1joint": -0.533013847,
  "lwheel": 0.0,
  "ljoint4": -0.568,
  "llink4child1joint": 0.533013505,
  "llink4child2joint": -0.186027678,
  "llink4child3joint": 0.533014617,
  "rjoint1": -0.312,
  "rlink1child1joint": 0.533013847,
  "rwheel": 0.0,
  "rjoint4": 0.568,
  "rlink4child1joint": -0.533013505,
  "rlink4child2joint": 0.186027678,
  "rlink4child3joint": -0.533014617,
}
MOTOR_ZERO_RAD = (-0.06, -0.20, 0.0, 0.06, 0.20, 0.0)
TORQUE_LIMITS = (40.0, 40.0, 5.2, 40.0, 40.0, 5.2)
LEG_LINK_1 = 0.220
LEG_LINK_2 = 0.260
WHEEL_RADIUS = 0.060
IMU_OFFSET = (0.2, 0.0, 0.0)


def load_wbr_spec() -> mujoco.MjSpec:
  """Load the collision-only, self-contained WBR MJCF."""
  return mujoco.MjSpec.from_file(str(WBR_XML_PATH))


def _load_wbr_entity_spec() -> mujoco.MjSpec:
  spec = load_wbr_spec()
  # EntityCfg writes an equivalent float32-safe init_state keyframe from
  # HOME_JOINT_POS. Remove the source key only from the composed runtime spec.
  for key in tuple(spec.keys):
    spec.delete(key)
  return spec


def get_wbr_robot_cfg() -> EntityCfg:
  """Create the mjlab entity config while retaining all XML actuators."""
  return EntityCfg(
    spec_fn=_load_wbr_entity_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.175),
      rot=(1.0, 0.0, 0.0, 0.0),
      # Explicitly retain every passive closed-chain coordinate from the MJCF home
      # keyframe. mjlab's joint_pos=None path keeps MuJoCo float64 tensors, while
      # MuJoCo Warp state is float32; the explicit map preserves the pose and dtype.
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
