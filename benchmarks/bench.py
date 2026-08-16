"""Throughput benchmark grid: batch size x resolution.

Run on an otherwise-idle machine. Uses the same primitives scene as the
tests (pass a model path as argv[1] to benchmark your own scene).

Usage: python benchmarks/bench.py [scene.xml]
"""
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "src"))
from mjbatch import BatchRenderer

SCENE = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3"><joint type="free"/>
      <geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.15 0.1 0.2"><joint type="free"/>
      <geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
  </worldbody>
</mujoco>
"""

if len(sys.argv) > 1:
    model = mujoco.MjModel.from_xml_path(sys.argv[1])
    cam = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
else:
    model = mujoco.MjModel.from_xml_string(SCENE)
    cam = "cam"
cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)

print(f"| N envs | res | batches/s | env-frames/s |")
print(f"|---|---|---|---|")
for res in (64, 128, 256):
    for n in (16, 64, 256, 1024):
        if n * res * res > 1024 * 256 * 256:
            continue                      # atlas cap for this grid
        r = BatchRenderer(model, n, width=res, height=res, camera=cam)
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        datas = [d] * n
        r.set_backgrounds(range(n), [np.zeros((res, res, 3), np.uint8)] * n)
        args = (datas, np.tile(model.cam_pos[cid], (n, 1)),
                np.tile(model.cam_quat[cid], (n, 1)))
        for _ in range(3):
            r.render(*args)
        reps = max(10, min(200, int(3000 / max(n // 16, 1))))
        t0 = time.time()
        for _ in range(reps):
            r.render(*args)
        dt = time.time() - t0
        print(f"| {n} | {res} | {reps/dt:.0f} | {n*reps/dt:,.0f} |", flush=True)
        del r
