"""Parity tests against mujoco.Renderer (the credibility anchor).

Renders random states of a primitives-only scene with both renderers and
checks foreground-silhouette agreement (IoU). Color agreement is NOT tested
exactly: shading models differ by design (Lambert vs MuJoCo's lighting).
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
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/>
    </body>
    <body pos="0.15 0.1 0.2">
      <joint type="free"/>
      <geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
    </body>
    <body pos="-0.15 -0.1 0.25">
      <joint type="free"/>
      <geom name="c1" type="cylinder" size="0.04 0.08" rgba="0.2 0.2 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(SCENE)


def _foreground_iou(model, seed):
    d = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    d.qpos[:] += rng.uniform(-0.05, 0.05, model.nq)
    mujoco.mj_forward(model, d)

    r = BatchRenderer(model, 1, width=256, height=256, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    bg = np.full((256, 256, 3), 120, np.uint8)
    r.set_backgrounds([0], [bg])
    ours = r.render([d], model.cam_pos[cid][None], model.cam_quat[cid][None])[0]

    ref_r = mujoco.Renderer(model, 256, 256)
    seg_r = mujoco.Renderer(model, 256, 256)
    seg_r.enable_segmentation_rendering()
    ref_r.update_scene(d, camera="cam")
    seg_r.update_scene(d, camera="cam")
    seg = seg_r.render()
    mask_ref = np.isin(seg[:, :, 0].astype(int), r.geoms)
    mask_ours = np.any(np.abs(ours.astype(int) - bg.astype(int)) > 8, axis=2)
    inter = (mask_ref & mask_ours).sum()
    union = (mask_ref | mask_ours).sum()
    return inter / max(union, 1)


def test_silhouette_parity(model):
    ious = [_foreground_iou(model, s) for s in range(4)]
    assert min(ious) > 0.90, f"silhouette IoU too low: {ious}"


def test_batch_tiles_shape(model):
    n = 9
    r = BatchRenderer(model, n, width=96, height=96, camera="cam")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    datas = [mujoco.MjData(model) for _ in range(n)]
    for d in datas:
        mujoco.mj_forward(model, d)
    r.set_backgrounds(range(n), [np.zeros((96, 96, 3), np.uint8)] * n)
    tiles = r.render(datas, np.tile(model.cam_pos[cid], (n, 1)),
                     np.tile(model.cam_quat[cid], (n, 1)))
    assert tiles.shape == (n, 96, 96, 3)
    assert tiles.dtype == np.uint8


def test_backgrounds_composited(model):
    r = BatchRenderer(model, 1, width=128, height=128, camera="cam", include_planes=False)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    bg = np.full((128, 128, 3), (200, 10, 250), np.uint8)
    r.set_backgrounds([0], [bg])
    out = r.render([d], model.cam_pos[cid][None], model.cam_quat[cid][None])[0]
    # corners should be background (no geometry there)
    assert (np.abs(out[0, 0].astype(int) - [200, 10, 250]) < 12).all()
