# Benchmark results — Apple M4 Max (40-core GPU), macOS

Measured on an otherwise-idle machine (concurrent jobs SIGSTOPped), wgpu
Metal backend, primitives test scene (3 geoms + plane; `benchmarks/bench.py`
with no arguments). Scene complexity matters: see the high-complexity row
below for a real robot scene.

| N envs | res | batches/s | env-frames/s |
|---|---|---|---|
| 16 | 64 | 518 | 8,288 |
| 64 | 64 | 335 | 21,438 |
| 256 | 64 | 124 | 31,731 |
| 1024 | 64 | 34 | 34,375 |
| 16 | 128 | 388 | 6,213 |
| 64 | 128 | 190 | 12,142 |
| 256 | 128 | 53 | 13,531 |
| 1024 | 128 | 11 | 11,192 |
| 16 | 256 | 223 | 3,572 |
| 64 | 256 | 64 | 4,070 |
| 256 | 256 | 15 | 3,936 |
| 1024 | 256 | 3 | 3,157 |

High-complexity scene (SO-101 arm, UNDECIMATED, 348k faces -> 162k indexed
verts): **8,365 env-frames/s** at 128 px, N=64. (The earlier unindexed
architecture managed 2,540 on this scene and needed mesh decimation to reach
6,000; indexed unique-mesh instancing made decimation optional.)
Texture parity: grayscale pattern correlation vs mujoco.Renderer on a
checkerboard+gradient scene = **0.994**.


## Context vs madrona_mjx

madrona_mjx reports ~403,000 frames/s at 64x64 on an NVIDIA RTX 4090
(as measured in the PyBatchRender paper's comparison, arXiv:2601.01288).
mjbatch-metal reaches ~34,000 frames/s at the same resolution on an M4 Max
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
| textures (MuJoCo materials, mesh UVs, planes/boxes) | yes | yes — atlas-packed, parity 0.994 vs mujoco.Renderer (`test_textures.py`) |
| high-complexity scenes | yes (~30K fps @ 7M tris, RTX 4090) | indexed unique-mesh instancing; 8.4K env-fps on a 348K-tri scene (M4 Max); no culling/LOD yet |
| platform | Linux + NVIDIA | macOS + Apple Silicon |

Fidelity context: neither renderer targets photorealism — both are
throughput-first "pixels to actions" renderers; photorealism is delegated to
a second engine in both ecosystems (Isaac RTX/Omniverse there, Unreal here).
With textures and indexed high-complexity geometry, mjbatch-metal now covers
Madrona's fidelity tier (minus shadows); remaining gaps are performance
architecture (culling/LOD for multi-million-triangle scenes) rather than
feature class.

Determinism contract: identical calls are byte-identical; identical content
in different atlas tiles is identical except at z-fighting pixels of
interpenetrating geometry (measured ~0.01% of pixels on a stress scene).
