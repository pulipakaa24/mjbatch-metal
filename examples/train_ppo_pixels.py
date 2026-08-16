"""PPO from pixels on the Metal batch backend (BatchPixelVecEnv).

Same recipe as train_rl_pixels.py (lerobot-sim2real port) but ~12x faster:
threaded physics + one batched Metal render per step.

Usage: KMP_DUPLICATE_LIB_OK=TRUE uv run --no-sync python train_rl_pixels_batch.py [steps]
Env: N_ENVS (64), OUT_DIR, BG_DIR / CAM_POS / CAM_QUAT for arrival-day setup.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np
from so101_lift_vecenv import BatchPixelVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

RUN = Path(os.environ.get("OUT_DIR", "/Users/adipu/so101Sim/rl/runs/lift_pixel_rl_batch"))
RUN.mkdir(parents=True, exist_ok=True)
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 25_000_000
N_ENVS = int(os.environ.get("N_ENVS", 64))
CKPT_EVERY = 2_000_000


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()
        self._t0 = time.time()
        self._f = open(RUN / "progress.txt", "a", buffering=1)
        self._next = CKPT_EVERY

    def _on_step(self) -> bool:
        if self.n_calls % 200 == 0:
            buf = self.model.ep_info_buffer
            if buf:
                sr = float(np.mean([e.get("is_success", 0.0) for e in buf]))
                rew = float(np.mean([e["r"] for e in buf]))
                fps = int(self.num_timesteps / max(1e-9, time.time() - self._t0))
                self._f.write(f"steps={self.num_timesteps} success={sr:.2f} "
                              f"ep_rew={rew:.1f} fps={fps}\n")
        if self.num_timesteps >= self._next:
            self.model.save(RUN / f"ckpt_{self._next // 1_000_000}M")
            self._f.write(f"ckpt saved at {self.num_timesteps}\n")
            self._next += CKPT_EVERY
        return True


if __name__ == "__main__":
    venv = BatchPixelVecEnv(n_envs=N_ENVS, seed=0)
    model = PPO(
        "MultiInputPolicy", venv,
        learning_rate=3e-4,
        n_steps=64,                # 64 envs x 64 = 4096 buffer
        batch_size=512,
        n_epochs=8,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        device="mps",
        verbose=1,
    )
    model.learn(total_timesteps=TOTAL, callback=Progress(), progress_bar=False)
    model.save(RUN / "model")
    print("done")
