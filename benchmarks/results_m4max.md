# Benchmark results — Apple M4 Max (40-core GPU), macOS

Measured on an otherwise-idle machine (concurrent jobs SIGSTOPped), wgpu
Metal backend, primitives test scene (3 geoms + plane; `benchmarks/bench.py`
with no arguments). Scene complexity matters: the SO-101 arm scene
(20 visual geoms, decimated meshes) reaches ~6,000-6,800 env-frames/s at
128 px where this scene reaches ~12,000-17,000.

| N envs | res | batches/s | env-frames/s |
|---|---|---|---|
| 16 | 64 | 516 | 8,254 |
| 64 | 64 | 336 | 21,504 |
| 256 | 64 | 147 | 37,565 |
| 1024 | 64 | 45 | 46,047 |
| 16 | 128 | 400 | 6,399 |
| 64 | 128 | 200 | 12,818 |
| 256 | 128 | 64 | 16,327 |
| 1024 | 128 | 17 | 17,425 |
| 16 | 256 | 219 | 3,511 |
| 64 | 256 | 71 | 4,516 |
| 256 | 256 | 19 | 4,873 |
| 1024 | 256 | 5 | 4,963 |


## Context vs madrona_mjx

madrona_mjx reports ~403,000 frames/s at 64x64 on an NVIDIA RTX 4090
(as measured in the PyBatchRender paper's comparison, arXiv:2601.01288).
mjbatch-metal reaches ~46,000 frames/s at the same resolution on an M4 Max
laptop — roughly an order of magnitude less throughput on hardware with
roughly an order of magnitude less rendering horsepower, and with no CUDA,
no Linux, and no discrete GPU. The point is not to beat a 4090; it is that
batch rendering at RL-useful rates exists on Apple Silicon at all.

## Feature coverage vs madrona_mjx (tests in `tests/`)

| capability | madrona_mjx | mjbatch-metal |
|---|---|---|
| batched tiled rendering, single submission | yes | yes (`test_large_batch_1024`) |
| RGB observations | yes | yes |
| depth observations | yes | yes, metric, parity-tested vs mujoco.Renderer (`test_depth_parity_vs_mujoco`, median <1 cm) |
| per-env domain randomization (color/light/camera) | yes | yes (`test_per_env_dr_differentiation`) |
| deterministic output | - | yes, byte-identical (`test_determinism`) |
| in-shader background compositing | no (post-hoc) | yes |
| GPU-resident physics (MJX) | yes | no — physics is CPU (MuJoCo C, threaded); on Apple Silicon CPU physics is not the bottleneck |
| CUDA graphs / JAX integration | yes | no |
| textured / high-complexity scenes (e.g. Habitat, ~7M tris) | yes (~30K fps) | no — untextured Lambert shading; realism comes from compositing real photos instead |
| platform | Linux + NVIDIA | macOS + Apple Silicon |

Fidelity context: neither renderer targets photorealism — Madrona's stated
purpose is high-throughput "pixels to actions" training, and its metrics are
framerates, not visual quality. The fidelity gradient is: mjbatch-metal
(flat-shaded, composited) < Madrona (textured complex scenes, still
throughput-first) < photoreal engines (Isaac RTX/Omniverse, Unreal). Both
ecosystems delegate photorealism to a second renderer; this project pairs
with Unreal Engine for that tier.
