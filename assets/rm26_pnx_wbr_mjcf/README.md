# RM26 PNX WBR model

Vendored from https://github.com/CosmosMount/rm26_pnx_wbr_mjcf at commit
`9c09472d3b714dbc0d7e8bacf9ac4bcafc31fd25` (2026-08-27 retrieval).
`UPSTREAM.json` records the upstream `mjmodel.xml` and 15 referenced STL SHA-256
hashes. The local XML may intentionally customize its `home` keyframe; the mesh
files remain byte-for-byte upstream copies. The unreferenced 98 MB assembly STL
and URDF are not needed by this MJCF and are not copied.

No license file was present at this revision. Upstream authors retain their
rights; check redistribution permission before publishing these assets.

## Runtime adaptations in `src/wbr_mjlab/robot.py`

- Give the 15 unnamed mesh geoms stable names for mjlab contact sensors. All
  non-wheel geoms participate in the terrain-contact penalty (13 instead of 5).
- Set robot `contype=2`, `conaffinity=1`, preserving the existing task's
  terrain-only collision policy. At the task reset pose the unfiltered upstream
  model has closed-chain pivot convex-hull penetration of up to 17 mm. Enabling
  robot self-collision requires decomposed collision assets or validated pair
  exclusions first. Mesh-to-terrain collisions remain enabled; no primitive
  box/capsule robot geoms replace the meshes.
- Treat the XML `home` keyframe as the source of truth for root position,
  orientation, all joint defaults, policy references and reset state. For mjlab
  composition, these XML values are parsed into an equivalent float32 entity init
  state; native MuJoCo retains and resets the original keyframe directly.

Body inertias, joint axes/limits/damping, mesh vertices, friction, sensors,
equality constraints, tendons and actuator definitions are retained. The control
clamp follows upstream motor limits: 20 Nm for legs, 5.2 Nm for wheels. The
upstream `ljoint4` joint-level cap is 40 Nm but its actuator cap is 20 Nm, so the
effective limit is still 20 Nm. The IMU site is now at `(0, 0, 0)`.

The observation/action dimensions are unchanged, so old policy weights can be
loaded. Their motion and rewards are not equivalent: contact geometry, contact
penalty coverage and motor limits differ from the old simplified model. Evaluate
before resuming training, and keep results in a separate run.

MuJoCo uses convex collision handling for these mesh geoms; rendering shows the
full triangle meshes. See the [MuJoCo collision documentation](https://mujoco.readthedocs.io/en/latest/computation.html#collision-detection).
