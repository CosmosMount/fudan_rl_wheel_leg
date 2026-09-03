"""Static MuJoCo XML terrain loading shared by mjlab and native simulation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.entity import Entity, EntityCfg

from .robot import PROJECT_ROOT

TERRAIN_XML_ENV = "WBR_TERRAIN_XML"


def resolve_terrain_xml(path: str | Path) -> Path:
  """Resolve an XML path, treating relative paths as project-root relative."""
  resolved = Path(path).expanduser()
  if not resolved.is_absolute():
    resolved = PROJECT_ROOT / resolved
  resolved = resolved.resolve()
  if not resolved.is_file():
    raise FileNotFoundError(f"Terrain XML does not exist: {resolved}")
  if resolved.suffix.lower() not in (".xml", ".mjcf"):
    raise ValueError(f"Terrain file must be .xml or .mjcf: {resolved}")
  return resolved


def configured_terrain_xml(path: str | Path | None = None) -> Path | None:
  """Return an explicit terrain path or the path selected through the environment."""
  selected = path if path is not None else os.environ.get(TERRAIN_XML_ENV)
  return None if selected in (None, "") else resolve_terrain_xml(selected)


def load_terrain_spec(path: str | Path) -> mujoco.MjSpec:
  """Load and validate a static terrain MJCF, wrapped in one fixed root body."""
  xml_path = resolve_terrain_xml(path)
  source = mujoco.MjSpec.from_file(str(xml_path))
  if source.joints:
    names = ", ".join(joint.name or "<unnamed>" for joint in source.joints)
    raise ValueError(f"Terrain XML must be static and contain no joints: {names}")
  if source.actuators:
    names = ", ".join(actuator.name or "<unnamed>" for actuator in source.actuators)
    raise ValueError(f"Terrain XML must contain no actuators: {names}")
  if not source.geoms:
    raise ValueError(f"Terrain XML contains no geoms: {xml_path}")

  # Contact sensors and indexing require names. Preserve supplied names and give
  # deterministic names only to anonymous geoms.
  used_names = {geom.name for geom in source.geoms if geom.name}
  for index, geom in enumerate(source.geoms):
    if geom.name:
      continue
    candidate = f"terrain_{index}"
    suffix = 1
    while candidate in used_names:
      candidate = f"terrain_{index}_{suffix}"
      suffix += 1
    geom.name = candidate
    used_names.add(candidate)

  # A terrain is scene geometry, not an independently resettable object. Wrap
  # world-level geoms in a fixed body so mjlab can index it as a regular Entity
  # without converting it to mocap.
  for key in tuple(source.keys):
    source.delete(key)
  wrapped = mujoco.MjSpec()
  root = wrapped.worldbody.add_body(name="xml_root")
  wrapped.attach(source, prefix="", frame=root.add_frame())
  return wrapped


class XmlTerrainEntity(Entity):
  """Entity variant that keeps XML terrain fixed instead of auto-wrapping as mocap."""

  cfg: XmlTerrainEntityCfg

  def _build_spec(self) -> None:
    self._spec = load_terrain_spec(self.cfg.xml_path)

  def _add_initial_state_keyframe(self) -> None:
    pass


@dataclass
class XmlTerrainEntityCfg(EntityCfg):
  """Configuration for a static terrain loaded from a MuJoCo XML file."""

  xml_path: str = ""

  def build(self) -> XmlTerrainEntity:
    return XmlTerrainEntity(self)


def get_xml_terrain_cfg(path: str | Path) -> XmlTerrainEntityCfg:
  return XmlTerrainEntityCfg(xml_path=str(resolve_terrain_xml(path)))


def attach_xml_terrain(spec: mujoco.MjSpec, path: str | Path) -> tuple[str, ...]:
  """Attach XML terrain to a native model and return its compiled geom names."""
  terrain = load_terrain_spec(path)
  local_geom_names = tuple(geom.name for geom in terrain.geoms)
  spec.attach(terrain, prefix="terrain/", frame=spec.worldbody.add_frame())
  return tuple(f"terrain/{name}" for name in local_geom_names)
