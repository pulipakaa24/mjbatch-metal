"""Texture support tests: MuJoCo textures render, pattern-match the
reference renderer, and the many-unique-meshes draw path is exercised.
"""
import mujoco
import numpy as np
import pytest

from mjbatch import BatchRenderer

TEX_SCENE = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.1 0.1 0.1"
             rgb2="0.95 0.95 0.95" width="64" height="64"/>
    <material name="floor_mat" texture="checker" texrepeat="6 6"/>
    <texture name="grad" type="2d" builtin="gradient" rgb1="1 0 0" rgb2="0 0 1"
             width="32" height="32"/>
    <material name="box_mat" texture="grad"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.9 -0.9 0.8" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.1" material="floor_mat"/>
    <body pos="0 0 0.25"><joint type="free"/>
      <geom name="b1" type="box" size="0.12 0.1 0.08" material="box_mat"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(TEX_SCENE)


def _render_both(model, res=256):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    r = BatchRenderer(model, 1, width=res, height=res, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    ours = r.render([d], model.cam_pos[cid][None], model.cam_quat[cid][None])[0]
    ref_r = mujoco.Renderer(model, res, res)
    ref_r.update_scene(d, camera="cam")
    ref = ref_r.render()
    return ours, ref


def test_texture_pattern_visible(model):
    """The checker floor must produce real spatial variance (not flat color)."""
    ours, _ = _render_both(model)
    lower = ours[170:250, 40:210].astype(float).mean(axis=2)  # floor region
    assert lower.std() > 25, f"floor texture variance too low: {lower.std():.1f}"


def test_texture_pattern_correlates_with_reference(model):
    """Grayscale pattern correlates with mujoco.Renderer's render.

    Shading models differ, so require moderate (not perfect) correlation —
    an untextured render correlates near zero against the checkerboard.
    """
    ours, ref = _render_both(model)
    a = ours.astype(float).mean(axis=2).ravel()
    b = ref.astype(float).mean(axis=2).ravel()
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.55, f"pattern correlation with reference too low: {r:.3f}"


def test_many_unique_meshes():
    """Scene with many distinct primitives exercises per-mesh instanced draws."""
    rng = np.random.default_rng(0)
    bodies = "\n".join(
        f'<body pos="{rng.uniform(-1,1):.3f} {rng.uniform(-1,1):.3f} {rng.uniform(0.1,0.8):.3f}">'
        f'<joint type="free"/>'
        f'<geom type="{["box","sphere","cylinder","capsule"][i % 4]}" '
        f'size="{rng.uniform(0.03,0.1):.4f} {rng.uniform(0.03,0.1):.4f} {rng.uniform(0.03,0.1):.4f}" '
        f'rgba="{rng.uniform(0.2,1):.2f} {rng.uniform(0.2,1):.2f} {rng.uniform(0.2,1):.2f} 1"/></body>'
        for i in range(40))
    xml = f"""
    <mujoco><worldbody>
      <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
      <camera name="cam" pos="0 -2.5 1.8" quat="0.93 0.36 0 0"/>
      {bodies}
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    r = BatchRenderer(m, 16, width=128, height=128, camera="cam")
    assert len(r._draws) >= 30, f"expected many unique meshes, got {len(r._draws)}"
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = 0
    tiles = r.render([d] * 16, np.tile(m.cam_pos[cid], (16, 1)),
                     np.tile(m.cam_quat[cid], (16, 1)))
    assert tiles.shape == (16, 128, 128, 3)
    # cross-tile equality holds except at z-fighting pixels: this scene has
    # interpenetrating bodies (no settling), and coplanar intersections
    # resolve differently under per-tile sub-pixel offsets (measured ~0.01%)
    frac_diff = (tiles[0] != tiles[-1]).any(axis=2).mean()
    assert frac_diff < 0.005, f"cross-tile difference too large: {frac_diff:.4%}"
    assert (tiles[0].sum(axis=2) > 30).sum() > 300
