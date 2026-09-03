from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from wbr_mjlab.sim2sim import NativeRunner, parse_args
from wbr_mjlab.task import make_env_cfg
from wbr_mjlab.terrain import load_terrain_spec, resolve_terrain_xml


def _write_static_terrain(path: Path) -> Path:
  path.write_text(
    """<mujoco model="test_terrain">
  <asset>
    <material name="ground_mat" rgba="0.2 0.4 0.2 1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 0.05"
          friction="0.7 0.01 0.001" material="ground_mat"/>
    <body name="step_body" pos="1 0 0.05">
      <geom name="step" type="box" size="0.3 1 0.05" friction="0.9 0.02 0.002"/>
    </body>
  </worldbody>
</mujoco>
"""
  )
  return path


def test_xml_terrain_loads_in_mjlab_and_native(tmp_path: Path) -> None:
  from mjlab.envs import ManagerBasedRlEnv

  terrain_xml = _write_static_terrain(tmp_path / "terrain.xml")
  cfg = make_env_cfg("plane", play=True, terrain_xml=terrain_xml)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device="cpu")
  runner = NativeRunner(lambda obs, hist: np.zeros(6), "plane", terrain_xml=terrain_xml)
  try:
    env.reset(seed=1)
    assert cfg.scene.terrain is None
    assert tuple(cfg.scene.entities) == ("terrain", "robot")
    assert {"terrain/ground", "terrain/step"} <= {
      env.sim.mj_model.geom(i).name for i in range(env.sim.mj_model.ngeom)
    }
    assert {"terrain/ground", "terrain/step"} <= {
      runner.model.geom(i).name for i in runner.terrain_geom_ids
    }
    np.testing.assert_allclose(runner.model.geom("terrain/ground").friction, (0.7, 0.01, 0.001))
    np.testing.assert_allclose(runner.model.geom("terrain/step").friction, (0.9, 0.02, 0.002))
  finally:
    env.close()


def test_xml_terrain_path_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  terrain_xml = _write_static_terrain(tmp_path / "terrain.mjcf")
  monkeypatch.setenv("WBR_TERRAIN_XML", str(terrain_xml))
  assert make_env_cfg("jump", play=True).scene.terrain is None
  args = parse_args(["--onnx", "policy.onnx", "--terrain-xml", str(terrain_xml)])
  assert args.terrain_xml == terrain_xml.resolve()
  assert resolve_terrain_xml(terrain_xml) == terrain_xml.resolve()


def test_xml_terrain_rejects_dynamic_models(tmp_path: Path) -> None:
  terrain_xml = tmp_path / "dynamic.xml"
  terrain_xml.write_text(
    """<mujoco><worldbody><body><freejoint name="moving"/>
    <geom name="box" type="box" size="1 1 1"/></body></worldbody></mujoco>"""
  )
  with pytest.raises(ValueError, match="must be static"):
    load_terrain_spec(terrain_xml)


def test_xml_terrain_requires_geoms(tmp_path: Path) -> None:
  terrain_xml = tmp_path / "empty.xml"
  terrain_xml.write_text("<mujoco><worldbody/></mujoco>")
  with pytest.raises(ValueError, match="contains no geoms"):
    load_terrain_spec(terrain_xml)


def test_xml_terrain_resolves_relative_mesh_assets(tmp_path: Path) -> None:
  mesh_dir = tmp_path / "meshes"
  mesh_dir.mkdir()
  (mesh_dir / "wedge.obj").write_text(
    """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 -0.1
f 1 2 3
f 1 4 2
f 2 4 3
f 3 4 1
"""
  )
  terrain_xml = tmp_path / "mesh_terrain.xml"
  terrain_xml.write_text(
    """<mujoco>
  <compiler meshdir="meshes"/>
  <asset><mesh name="wedge" file="wedge.obj"/></asset>
  <worldbody><geom name="wedge_ground" type="mesh" mesh="wedge"/></worldbody>
</mujoco>
"""
  )
  runner = NativeRunner(lambda obs, hist: np.zeros(6), "plane", terrain_xml=terrain_xml)
  assert runner.model.geom("terrain/wedge_ground").type == mujoco.mjtGeom.mjGEOM_MESH
  assert runner.model.nmesh >= 16  # Robot meshes plus the imported terrain mesh.
