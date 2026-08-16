"""End-to-end throughput: threaded MuJoCo physics + batched pipelined render.

Measures the same quantity MuJoCo Playground reports for MJX+Madrona
("environment steps per second": physics + per-step observation rendering,
no learner), enabling a like-for-like methodology comparison — hardware
differences remain and are disclosed alongside results.

Scenes:
- built-in cartpole-class scene (2 geoms + rail), the analog of their
  CartpoleBalance figure
- pass any scene.xml as argv[1] for a robot-scale analog (e.g. an arm
  scene vs their PandaPickCubeCartesian)

Usage: python benchmarks/end_to_end.py [scene.xml] [n_envs] [res]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "src"))
from mjbatch import BatchRenderer

CARTPOLE = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.2 0.2 -0.95"/>
    <camera name="cam" pos="0 -2.2 0.7" quat="0.755 0.656 0 0"/>
    <geom name="rail" type="capsule" fromto="-1 0 0.4 1 0 0.4" size="0.02" rgba="0.6 0.6 0.6 1"/>
    <body pos="0 0 0.4">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="cart" type="box" size="0.1 0.06 0.04" rgba="0.2 0.4 0.9 1"/>
      <body pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" rgba="0.9 0.4 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slide" gear="50"/></actuator>
</mujoco>
"""


def bench(model, n, res, cam_name, steps=200, n_threads=12):
    datas = [mujoco.MjData(model) for _ in range(n)]
    for d in datas:
        mujoco.mj_forward(model, d)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    r = BatchRenderer(model, n, width=res, height=res, camera=cam_name)
    pool = ThreadPoolExecutor(max_workers=n_threads)
    rng = np.random.default_rng(0)
    ctrl = rng.uniform(-1, 1, (n, model.nu)) if model.nu else None

    def phys_chunk(chunk):
        for e in chunk:
            if ctrl is not None:
                datas[e].ctrl[:] = ctrl[e]
            mujoco.mj_step(model, datas[e])

    chunks = [range(i, n, n_threads) for i in range(n_threads)]
    cam_pos = np.tile(model.cam_pos[cid], (n, 1))
    cam_quat = np.tile(model.cam_quat[cid], (n, 1))

    def one_step():
        list(pool.map(phys_chunk, chunks))
        r.render(datas, cam_pos, cam_quat, pipelined=True)

    for _ in range(5):
        one_step()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    dt = time.time() - t0
    pool.shutdown(wait=False)
    return n * steps / dt


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".xml"):
        model = mujoco.MjModel.from_xml_path(sys.argv[1])
        cam = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        res = int(sys.argv[3]) if len(sys.argv) > 3 else 128
        print(f"| custom scene | N={n} @{res}px | {bench(model, n, res, cam):,.0f} steps/s |")
    else:
        model = mujoco.MjModel.from_xml_string(CARTPOLE)
        for n, res in ((1024, 64), (4096, 64)):
            print(f"| cartpole-class | N={n} @{res}px | {bench(model, n, res, 'cam'):,.0f} steps/s |",
                  flush=True)
