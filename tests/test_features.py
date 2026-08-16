"""Feature tests targeting madrona_mjx-class functionality:
depth output, determinism, per-env DR, large batches, multi-camera use.
"""
import mujoco
import numpy as np
import pytest

from mjbatch import BatchRenderer

SCENE = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <camera name="top" pos="0 0 1.2" quat="1 0 0 0"/>
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


def test_depth_parity_vs_mujoco(model):
    """Metric depth agrees with mujoco.Renderer's depth on foreground pixels."""
    d = _data(model)
    r = BatchRenderer(model, 1, width=256, height=256, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    r.set_backgrounds([0], [np.zeros((256, 256, 3), np.uint8)])
    tiles, depth = r.render([d], model.cam_pos[cid][None], model.cam_quat[cid][None],
                            return_depth=True)
    ref = mujoco.Renderer(model, 256, 256)
    ref.enable_depth_rendering()
    ref.update_scene(d, camera="cam")
    ref_depth = ref.render()
    # compare where both see foreground (near geometry, not floor/background)
    both = (depth[0] < 2.0) & (ref_depth < 2.0)
    assert both.sum() > 500, "no overlapping foreground"
    err = np.abs(depth[0][both] - ref_depth[both])
    assert np.median(err) < 0.01, f"median depth error {np.median(err):.4f} m"
    assert np.percentile(err, 90) < 0.05, f"p90 depth error {np.percentile(err,90):.4f} m"


def test_determinism(model):
    """Identical inputs produce byte-identical output."""
    d = _data(model)
    r = BatchRenderer(model, 4, width=128, height=128, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    r.set_backgrounds(range(4), [np.full((128, 128, 3), 90, np.uint8)] * 4)
    args = ([d] * 4, np.tile(model.cam_pos[cid], (4, 1)), np.tile(model.cam_quat[cid], (4, 1)))
    a = r.render(*args)
    b = r.render(*args)
    assert np.array_equal(a, b)


def test_per_env_dr_differentiation(model):
    """Per-env colors, camera jitter, and backgrounds actually differ per tile."""
    n = 4
    r = BatchRenderer(model, n, width=128, height=128, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    rng = np.random.default_rng(0)
    datas = [_data(model) for _ in range(n)]
    colors = rng.uniform(0.1, 1.0, (n, r.G, 4)).astype(np.float32)
    colors[..., 3] = 1
    cam_pos = np.tile(model.cam_pos[cid], (n, 1)) + rng.uniform(-0.03, 0.03, (n, 3))
    bgs = [np.full((128, 128, 3), c, np.uint8) for c in ((30, 30, 30), (200, 50, 50),
                                                         (50, 200, 50), (50, 50, 200))]
    r.set_backgrounds(range(n), bgs)
    tiles = r.render(datas, cam_pos, np.tile(model.cam_quat[cid], (n, 1)), colors)
    # every pair of tiles must differ substantially
    for i in range(n):
        for j in range(i + 1, n):
            assert np.abs(tiles[i].astype(int) - tiles[j].astype(int)).mean() > 5


def test_large_batch_1024(model):
    """1024 environments in one draw (madrona-scale batch)."""
    n = 1024
    r = BatchRenderer(model, n, width=64, height=64, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    d = _data(model)
    r.set_backgrounds(range(n), [np.zeros((64, 64, 3), np.uint8)] * n)
    tiles = r.render([d] * n, np.tile(model.cam_pos[cid], (n, 1)),
                     np.tile(model.cam_quat[cid], (n, 1)))
    assert tiles.shape == (n, 64, 64, 3)
    # all tiles identical inputs -> identical outputs; and non-empty
    assert np.array_equal(tiles[0], tiles[-1])
    assert (tiles[0].sum(axis=2) > 30).sum() > 50  # some foreground pixels


def test_two_cameras_same_env(model):
    """Multi-camera observation: render the same env from two cameras."""
    d = _data(model)
    r = BatchRenderer(model, 1, width=128, height=128, camera="cam", include_planes=False)
    c1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    c2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    r.set_backgrounds([0], [np.zeros((128, 128, 3), np.uint8)])
    a = r.render([d], model.cam_pos[c1][None], model.cam_quat[c1][None])
    b = r.render([d], model.cam_pos[c2][None], model.cam_quat[c2][None])
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 3  # genuinely different views


def test_stability_500_frames(model):
    """No degradation over sustained rendering."""
    import time
    n = 64
    r = BatchRenderer(model, n, width=128, height=128, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    datas = [_data(model) for _ in range(n)]
    r.set_backgrounds(range(n), [np.zeros((128, 128, 3), np.uint8)] * n)
    args = (datas, np.tile(model.cam_pos[cid], (n, 1)), np.tile(model.cam_quat[cid], (n, 1)))
    r.render(*args)
    t0 = time.time()
    first = None
    for i in range(500):
        if i == 50:
            first = time.time() - t0
        r.render(*args)
    total = time.time() - t0
    late_rate = 450 / (total - first)
    early_rate = 50 / first
    assert late_rate > 0.5 * early_rate, "throughput degraded >2x over 500 frames"
