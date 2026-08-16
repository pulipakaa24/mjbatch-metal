"""Generate the README gallery images (reproducible).

Usage: python benchmarks/make_gallery.py [robot_scene.xml]
Writes PNGs into assets/. Without a robot scene argument, the robot-scene
images are skipped and only the built-in scenes are rendered.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mjbatch import BatchRenderer

ASSETS = Path(__file__).parents[1] / "assets"
ASSETS.mkdir(exist_ok=True)


def save(name, img):
    from PIL import Image
    Image.fromarray(img).save(ASSETS / name)
    print("wrote", ASSETS / name)


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
    <body pos="0.35 0.3 0.2"><joint type="free"/>
      <geom type="sphere" size="0.08" rgba="0.2 0.8 0.3 1"/></body>
  </worldbody>
</mujoco>
"""

CARTPOLE = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.2 0.2 -0.95"/>
    <camera name="cam" pos="0 -2.2 0.7" quat="0.755 0.656 0 0"/>
    <geom name="rail" type="capsule" fromto="-1 0 0.4 1 0 0.4" size="0.02" rgba="0.6 0.6 0.6 1"/>
    <body pos="0 0 0.4">
      <joint type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.06 0.04" rgba="0.2 0.4 0.9 1"/>
      <body pos="0 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" rgba="0.9 0.4 0.2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def textured_scene_images():
    m = mujoco.MjModel.from_xml_string(TEX_SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    r = BatchRenderer(m, 1, width=256, height=256, camera="cam")
    ours, depth, seg = r.render([d], m.cam_pos[cid][None], m.cam_quat[cid][None],
                                return_depth=True, return_seg=True)
    ref = mujoco.Renderer(m, 256, 256)
    ref.update_scene(d, camera="cam")
    save("tex_parity.png", np.hstack([ref.render(), ours[0]]))
    # depth colormap (near=dark, far=light, clipped to 3m)
    dnorm = np.clip(depth[0] / 3.0, 0, 1)
    dimg = (np.stack([dnorm]*3, axis=2) * 255).astype(np.uint8)
    # segmentation as distinct colors
    palette = np.array([[30, 30, 30], [230, 80, 80], [80, 200, 90],
                        [90, 130, 240], [240, 200, 70]], np.uint8)
    simg = palette[seg[0] % len(palette)]
    save("rgb_depth_seg.png", np.hstack([ours[0], dimg, simg]))


def cartpole_atlas():
    m = mujoco.MjModel.from_xml_string(CARTPOLE)
    n = 64
    r = BatchRenderer(m, n, width=64, height=64, camera="cam")
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    rng = np.random.default_rng(3)
    datas = []
    for i in range(n):
        d = mujoco.MjData(m)
        d.qpos[0] = rng.uniform(-0.6, 0.6)
        d.qpos[1] = rng.uniform(-0.9, 0.9)
        mujoco.mj_forward(m, d)
        datas.append(d)
    colors = np.zeros((n, r.G, 4), np.float32)
    base = r.default_colors()
    for e in range(n):
        colors[e] = base
        colors[e, :, :3] = np.clip(base[:, :3] + rng.uniform(-0.25, 0.25, (r.G, 3)), 0, 1)
    cam_pos = np.tile(m.cam_pos[cid], (n, 1)) + rng.uniform(-0.05, 0.05, (n, 3))
    tiles = r.render(datas, cam_pos, np.tile(m.cam_quat[cid], (n, 1)), colors)
    atlas = np.vstack([np.hstack(tiles[i*8:(i+1)*8]) for i in range(8)])
    save("cartpole_atlas_64.png", atlas)


def robot_scene_images(path):
    m = mujoco.MjModel.from_xml_path(path)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "base_cam")
    cam = "base_cam" if cid >= 0 else mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, 0)
    cid = max(cid, 0)
    n = 16
    r = BatchRenderer(m, n, width=128, height=128, camera=cam,
                      include_planes=False, decimate_faces=0)
    rng = np.random.default_rng(5)
    datas = []
    for i in range(n):
        d = mujoco.MjData(m)
        kf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "pickup")
        if kf >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kf)
        d.qpos[:6] += rng.uniform(-0.4, 0.4, 6)
        mujoco.mj_forward(m, d)
        datas.append(d)
    # DR: colors + procedural backgrounds
    base = r.default_colors()
    colors = np.zeros((n, r.G, 4), np.float32)
    bgs = []
    for e in range(n):
        c = np.clip(rng.uniform(0.1, 0.95, 3), 0, 1)
        colors[e] = base
        colors[e, :, :3] = np.clip(c[None, :] + rng.uniform(-0.12, 0.12, (r.G, 3)), 0, 1)
        kind = e % 3
        if kind == 0:
            bg = np.full((128, 128, 3), rng.integers(0, 256, 3), np.uint8)
        elif kind == 1:
            a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            t = np.linspace(0, 1, 128)[:, None, None]
            bg = (a * (1 - t) + b * t).astype(np.uint8)
            bg = np.broadcast_to(bg, (128, 128, 3)).copy()
        else:
            sm = rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
            idx = np.linspace(0, 7, 128).astype(int)
            bg = sm[np.ix_(idx, idx)]
        bgs.append(bg)
    r.set_backgrounds(range(n), bgs)
    cam_pos = np.tile(m.cam_pos[cid], (n, 1)) + rng.uniform(-0.015, 0.015, (n, 3))
    tiles = r.render(datas, cam_pos, np.tile(m.cam_quat[cid], (n, 1)), colors)
    grid = np.vstack([np.hstack(tiles[i*4:(i+1)*4]) for i in range(4)])
    save("robot_dr_grid.png", grid)
    # complexity ladder: cartpole tile | textured scene | robot scene
    mt = mujoco.MjModel.from_xml_string(TEX_SCENE)
    dt_ = mujoco.MjData(mt); mujoco.mj_forward(mt, dt_)
    rt = BatchRenderer(mt, 1, width=128, height=128, camera="cam")
    tcid = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    tex_tile = rt.render([dt_], mt.cam_pos[tcid][None], mt.cam_quat[tcid][None])[0]
    mc = mujoco.MjModel.from_xml_string(CARTPOLE)
    dc = mujoco.MjData(mc); mujoco.mj_forward(mc, dc)
    rc = BatchRenderer(mc, 1, width=128, height=128, camera="cam")
    ccid = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    cart_tile = rc.render([dc], mc.cam_pos[ccid][None], mc.cam_quat[ccid][None])[0]
    save("complexity_ladder.png", np.hstack([cart_tile, tex_tile, tiles[0]]))


if __name__ == "__main__":
    textured_scene_images()
    cartpole_atlas()
    if len(sys.argv) > 1:
        robot_scene_images(sys.argv[1])
    print("gallery done")
