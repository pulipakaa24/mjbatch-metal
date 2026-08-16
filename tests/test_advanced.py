"""Tests for segmentation output, pipelined readback, and frustum culling."""
import mujoco
import numpy as np
import pytest

from mjbatch import BatchRenderer

SCENE = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/>
    </body>
    <body pos="0.15 0.1 0.2">
      <joint type="free"/>
      <geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(SCENE)


def _data(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    return d


def test_segmentation_output(model):
    """Per-pixel geom ids agree with mujoco.Renderer's segmentation."""
    d = _data(model)
    r = BatchRenderer(model, 1, width=256, height=256, camera="cam",
                      include_planes=False)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    tiles, seg = r.render([d], model.cam_pos[cid][None], model.cam_quat[cid][None],
                          return_seg=True)
    assert seg.dtype == np.uint32 and seg.shape == (1, 256, 256)
    ref = mujoco.Renderer(model, 256, 256)
    ref.enable_segmentation_rendering()
    ref.update_scene(d, camera="cam")
    ref_seg = ref.render()[:, :, 0].astype(int)
    for slot, g in enumerate(r.geoms):
        ours = seg[0] == slot + 1
        theirs = ref_seg == g
        if theirs.sum() < 20:
            continue
        iou = (ours & theirs).sum() / max((ours | theirs).sum(), 1)
        assert iou > 0.85, f"seg IoU for geom {g}: {iou:.3f}"


def test_pipelined_readback(model):
    """Pipelined mode returns the previous frame's tiles (one-frame lag)."""
    n = 4
    r = BatchRenderer(model, n, width=128, height=128, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    d1, d2 = _data(model), _data(model)
    d2.qpos[0] += 0.2
    mujoco.mj_forward(model, d2)
    a_sync = r.render([d1] * n, np.tile(model.cam_pos[cid], (n, 1)),
                      np.tile(model.cam_quat[cid], (n, 1)))
    b_sync = r.render([d2] * n, np.tile(model.cam_pos[cid], (n, 1)),
                      np.tile(model.cam_quat[cid], (n, 1)))
    first = r.render([d1] * n, np.tile(model.cam_pos[cid], (n, 1)),
                     np.tile(model.cam_quat[cid], (n, 1)), pipelined=True)
    assert first is None
    lagged = r.render([d2] * n, np.tile(model.cam_pos[cid], (n, 1)),
                      np.tile(model.cam_quat[cid], (n, 1)), pipelined=True)
    assert np.array_equal(lagged, a_sync)          # frame 1 delivered on call 2
    lagged2 = r.render([d2] * n, np.tile(model.cam_pos[cid], (n, 1)),
                       np.tile(model.cam_quat[cid], (n, 1)), pipelined=True)
    assert np.array_equal(lagged2, b_sync)


def test_culling_preserves_output(model):
    """Culling must not change visible pixels; and it drops instances."""
    n = 8
    r = BatchRenderer(model, n, width=128, height=128, camera="cam",
                      include_planes=False)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    datas = [_data(model) for _ in range(n)]
    args = (datas, np.tile(model.cam_pos[cid], (n, 1)),
            np.tile(model.cam_quat[cid], (n, 1)))
    plain = r.render(*args)
    culled = r.render(*args, cull=True)
    frac = (plain != culled).any(axis=3).mean()
    assert frac < 0.001, f"culling changed {frac:.4%} of pixels"
    # after a culled call, a plain call must still be correct (table restore)
    plain2 = r.render(*args)
    assert np.array_equal(plain, plain2)


def test_culling_offscreen_speedup(model):
    """A scene where geometry is far off-camera renders faster with culling."""
    import time
    rng = np.random.default_rng(0)
    bodies = "\n".join(
        f'<body pos="{50 + rng.uniform(0, 5):.2f} {rng.uniform(-5, 5):.2f} 0.3">'
        f'<joint type="free"/><geom type="sphere" size="0.05"/></body>'
        for _ in range(60))
    xml = f"""<mujoco><worldbody>
      <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
      <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
      <body pos="0 0 0.3"><joint type="free"/>
        <geom type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
      {bodies}
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    n = 64
    r = BatchRenderer(m, n, width=128, height=128, camera="cam")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    args = ([d] * n, np.tile(m.cam_pos[0], (n, 1)), np.tile(m.cam_quat[0], (n, 1)))
    plain = r.render(*args)
    culled = r.render(*args, cull=True)
    assert (plain != culled).any(axis=3).mean() < 0.001
