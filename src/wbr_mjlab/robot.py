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

HOME_ROOT_POS = (0.0, 0.0, 0.175)
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
TORQUE_LIMITS = (20.0, 20.0, 5.2, 20.0, 20.0, 5.2)
LEG_LINK_1 = 0.220
LEG_LINK_2 = 0.260
WHEEL_RADIUS = 0.060
IMU_OFFSET = (0.0, 0.0, 0.0)


def load_wbr_spec() -> mujoco.MjSpec:
  """Load the upstream mesh model with the documented RL scene adaptations."""
  spec = mujoco.MjSpec.from_file(str(WBR_XML_PATH))
  for geom in spec.geoms:
    geom.name = _MESH_GEOM_NAMES[geom.meshname]
    # Retain terrain-only collision filtering from the task. Upstream CAD convex
    # hulls overlap at closed-chain pivots (up to 17 mm at the reset pose).
    geom.contype = 2
    geom.conaffinity = 1
  # The upstream demonstration key is translated below the training floor.
  # This closed-chain solution keeps the mesh wheels just above z=0 at reset.
  spec.key("home").qpos = (
    *HOME_ROOT_POS,
    1.0,
    0.0,
    0.0,
    0.0,
    *(HOME_JOINT_POS[joint.name] for joint in spec.joints if joint.name in HOME_JOINT_POS),
  )
  return spec


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
      pos=HOME_ROOT_POS,
      rot=(1.0, 0.0, 0.0, 0.0),
      # Explicitly retain every passive closed-chain coordinate from the task home
      # pose. mjlab's joint_pos=None path keeps MuJoCo float64 tensors, while
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
