"""BatchPixelVecEnv — N lift envs, threaded physics + one batched Metal render.

SB3 VecEnv-compatible. Replicates SO101LiftEnv's reward/termination and
SO101PixelEnv's DR (visual + physics) but vectorized:
- per-env MjModel copies (physics DR mutates model fields)
- physics stepped by a thread pool (mj_step releases the GIL)
- observations rendered by mjbatch.BatchRenderer in ONE draw, composited
Obs: {"image": (N,128,128,3) u8, "qpos": (N,6) f32}
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from mjbatch import BatchRenderer

TILE = 128


def _load_bg_dir():
    """Load background images from $BG_DIR (jpg/png), center-cropped to TILE."""
    import os
    from pathlib import Path
    bgs = []
    bg_dir = os.environ.get("BG_DIR")
    if bg_dir and Path(bg_dir).is_dir():
        import imageio.v2 as imageio
        import numpy as np
        for p in sorted(Path(bg_dir).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                im = imageio.imread(p)
                if im.ndim == 3 and im.shape[2] >= 3:
                    h, w = im.shape[:2]
                    sz = min(h, w)
                    im = im[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2, :3]
                    idx = np.linspace(0, sz-1, TILE).astype(int)
                    bgs.append(im[np.ix_(idx, idx)].astype(np.uint8))
    return bgs



SCENE = os.environ.get("SCENE_XML", "/Users/adipu/so101Sim/mujoco_menagerie/robotstudio_so101/scene_box_rl.xml")
# between-the-jaws grasp point in gripperframe-site coordinates
# (from scripted_expert.py keyframe calibration)
_TCP_LOCAL = __import__("numpy").array([-0.00739, -0.01061, 0.01278])
LIFT_Z = 0.10
SUCCESS_HOLD = 10
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper")


class BatchPixelVecEnv(VecEnv):
    def __init__(self, n_envs: int = 64, ctrl_dt: float = 0.02,
                 max_episode_steps: int = 300, seed: int = 0, n_threads: int = 12):
        self.n = n_envs
        self.models = [mujoco.MjModel.from_xml_path(SCENE) for _ in range(n_envs)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        m0 = self.models[0]
        self._n_sub = max(1, int(round(ctrl_dt / m0.opt.timestep)))
        self._max_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(max_workers=n_threads)

        rngs = m0.actuator_ctrlrange.copy()
        limited = m0.actuator_ctrllimited.astype(bool).reshape(-1)
        self._ctrl_lo = np.where(limited, rngs[:, 0], -np.pi)
        self._ctrl_hi = np.where(limited, rngs[:, 1], np.pi)
        self._grip_site = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._box_body = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._box_geom = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "box")
        box_jnt = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, "box")
        self._box_qadr = int(m0.jnt_qposadr[box_jnt])
        self._arm_qadr = np.array([int(m0.jnt_qposadr[mujoco.mj_name2id(
            m0, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
        self._home = m0.qpos0.copy()
        self._mass0 = float(m0.body_mass[self._box_body])
        self._fric0 = m0.geom_friction[self._box_geom].copy()
        self._cam_id = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_CAMERA, "base_cam")
        # grasp detection: cube in contact with BOTH jaws simultaneously
        # (recipe-faithful: ManiSkill's SO100GraspCube reward is grasp-conditioned)
        self._reward_mode = os.environ.get("REWARD_MODE", "vanilla")
        grip_body = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        jaw_body = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self._grip_geoms = frozenset(g for g in range(m0.ngeom) if m0.geom_bodyid[g] == grip_body)
        self._jaw_geoms = frozenset(g for g in range(m0.ngeom) if m0.geom_bodyid[g] == jaw_body)
        if os.environ.get("CAM_POS"):
            p = [float(x) for x in os.environ["CAM_POS"].split()]
            for m in self.models:
                m.cam_pos[self._cam_id] = p
        if os.environ.get("CAM_QUAT"):
            q = [float(x) for x in os.environ["CAM_QUAT"].split()]
            for m in self.models:
                m.cam_quat[self._cam_id] = q
        self._cam_pos0 = self.models[0].cam_pos[self._cam_id].copy()
        self._cam_quat0 = self.models[0].cam_quat[self._cam_id].copy()

        self.renderer = BatchRenderer(m0, n_envs, width=TILE, height=TILE, camera="base_cam", include_planes=False, use_backgrounds=True)
        self._backgrounds = _load_bg_dir()
        self.G = self.renderer.G
        self._box_slot = self.renderer.geoms.index(self._box_geom)

        self._colors = np.ones((self.n, self.G, 4), np.float32)
        self._cam_pos = np.tile(self._cam_pos0, (self.n, 1))
        self._cam_quat = np.tile(self._cam_quat0, (self.n, 1))
        self._lights = np.tile([0.3, 0.3, -0.9], (self.n, 1)).astype(np.float64)
        self._t = np.zeros(self.n, np.int64)
        self._held = np.zeros(self.n, np.int64)
        self._succ = np.zeros(self.n, bool)
        self._ep_rew = np.zeros(self.n)

        obs_space = spaces.Dict({
            "image": spaces.Box(0, 255, (TILE, TILE, 3), np.uint8),
            "qpos": spaces.Box(-np.inf, np.inf, (6,), np.float32),
        })
        act_space = spaces.Box(-1.0, 1.0, (m0.nu,), np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions = None

    # -- DR / reset -------------------------------------------------------

    def _bg_for(self):
        if self._backgrounds and self.rng.random() < 0.8:
            return self._backgrounds[self.rng.integers(0, len(self._backgrounds))]
        kind = self.rng.integers(0, 3)
        if kind == 0:
            return np.full((TILE, TILE, 3), self.rng.integers(0, 256, 3), np.uint8)
        if kind == 1:
            a = self.rng.integers(0, 256, 3).astype(np.float32)
            b = self.rng.integers(0, 256, 3).astype(np.float32)
            t = np.linspace(0, 1, TILE)[:, None, None]
            return np.broadcast_to((a * (1 - t) + b * t).astype(np.uint8), (TILE, TILE, 3)).copy()
        small = self.rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
        idx = np.linspace(0, 7, TILE).astype(int)
        return small[np.ix_(idx, idx)]

    def _reset_env(self, e: int):
        m, d = self.models[e], self.datas[e]
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._home
        r = self.rng.uniform(0.15, 0.26)
        th = self.rng.uniform(-0.7, 0.7)
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[self._box_qadr:self._box_qadr + 3] = (r * np.cos(th), r * np.sin(th), 0.03)
        d.qpos[self._box_qadr + 3:self._box_qadr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        d.qpos[self._arm_qadr[:5]] += self.rng.uniform(-0.08, 0.08, 5)
        m.body_mass[self._box_body] = self._mass0 * self.rng.uniform(0.7, 1.4)
        m.geom_friction[self._box_geom] = self._fric0 * self.rng.uniform(0.7, 1.3)
        mujoco.mj_forward(m, d)
        # visual DR
        base = self.rng.uniform(0.1, 0.95, 3)
        cols = np.clip(base[None, :] + self.rng.uniform(-0.12, 0.12, (self.G, 3)), 0, 1)
        self._colors[e, :, :3] = cols
        self._colors[e, self._box_slot, :3] = np.clip(
            np.array([0.1, 0.8, 0.15]) + self.rng.uniform(-0.1, 0.1, 3), 0, 1)
        self._cam_pos[e] = self._cam_pos0 + self.rng.uniform(-0.015, 0.015, 3)
        q = self._cam_quat0 + self.rng.uniform(-0.01, 0.01, 4)
        self._cam_quat[e] = q / np.linalg.norm(q)
        self._lights[e] = np.array([0.3, 0.3, -0.9]) + self.rng.uniform(-0.15, 0.15, 3)
        self.renderer.set_backgrounds([e], [self._bg_for()])
        self._t[e] = 0
        self._held[e] = 0
        self._succ[e] = False
        self._ep_rew[e] = 0.0

    # -- VecEnv API -------------------------------------------------------

    def reset(self):
        for e in range(self.n):
            self._reset_env(e)
        return self._obs()

    def _obs(self):
        imgs = self.renderer.render(self.datas, self._cam_pos, self._cam_quat,
                                    self._colors, self._lights)
        qpos = np.stack([d.qpos[self._arm_qadr] for d in self.datas]).astype(np.float32)
        return {"image": imgs, "qpos": qpos}

    def step_async(self, actions):
        self._actions = np.clip(actions, -1.0, 1.0)

    def _phys_chunk(self, chunk):
        for e in chunk:
            m, d = self.models[e], self.datas[e]
            d.ctrl[:] = self._ctrl_lo + 0.5 * (self._actions[e] + 1.0) * (self._ctrl_hi - self._ctrl_lo)
            for _ in range(self._n_sub):
                mujoco.mj_step(m, d)

    def _is_grasped(self, e: int) -> bool:
        d = self.datas[e]
        hit_grip = hit_jaw = False
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if g1 == self._box_geom:
                other = g2
            elif g2 == self._box_geom:
                other = g1
            else:
                continue
            if other in self._grip_geoms:
                hit_grip = True
            elif other in self._jaw_geoms:
                hit_jaw = True
            if hit_grip and hit_jaw:
                return True
        return False

    def step_wait(self):
        n_chunks = self.pool._max_workers
        chunks = [range(i, self.n, n_chunks) for i in range(n_chunks)]
        list(self.pool.map(self._phys_chunk, chunks))
        infos = [{} for _ in range(self.n)]
        self._t += 1
        # vectorized reward across all envs
        box = np.stack([d.xpos[self._box_body] for d in self.datas])           # (N,3)
        # reach target = calibrated TCP (between the jaws), NOT the raw
        # gripperframe site — the site is offset ~1.8 cm from the grasp point
        # (same bug that held the scripted expert at 2%; offset from its
        # keyframe calibration). Reward peak must coincide with graspability.
        grip = np.stack([d.site_xpos[self._grip_site]
                         + d.site_xmat[self._grip_site].reshape(3, 3) @ _TCP_LOCAL
                         for d in self.datas])                                 # (N,3)
        dist = np.linalg.norm(box - grip, axis=1)
        height = box[:, 2]
        lifted = height > LIFT_Z
        r_reach = 1.0 - np.tanh(10.0 * dist)
        r_lift = 5.0 * np.clip((height - 0.035) / (LIFT_Z - 0.035), 0.0, 1.0)
        if self._reward_mode == "grasped":
            grasped = np.array([self._is_grasped(e) for e in range(self.n)])
            # grasp bonus; lift/hold pay only while actually grasping
            rewards = r_reach + 0.5 * grasped + (r_lift + 3.0 * lifted) * grasped
            lifted = lifted & grasped
        else:
            rewards = r_reach + r_lift + 3.0 * lifted
        self._held = np.where(lifted, self._held + 1, 0)
        self._succ |= self._held >= SUCCESS_HOLD
        oob = np.linalg.norm(box[:, :2], axis=1) > 0.45
        trunc = self._t >= self._max_steps
        dones = oob | trunc
        self._ep_rew += rewards
        for e in np.flatnonzero(dones):
            infos[e]["episode"] = {"r": float(self._ep_rew[e]), "l": int(self._t[e])}
            infos[e]["is_success"] = bool(self._succ[e])
            infos[e]["TimeLimit.truncated"] = bool(trunc[e] and not oob[e])
        obs = self._obs()
        if dones.any():
            for e in np.flatnonzero(dones):
                infos[e]["terminal_observation"] = {
                    "image": obs["image"][e].copy(), "qpos": obs["qpos"][e].copy()}
                self._reset_env(e)
            obs2 = self._obs()
            for e in np.flatnonzero(dones):
                obs["image"][e] = obs2["image"][e]
                obs["qpos"][e] = obs2["qpos"][e]
        return obs, rewards, dones, infos

    # -- VecEnv boilerplate ----------------------------------------------

    def close(self):
        self.pool.shutdown(wait=False)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.n

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.n

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.n
