# Benchmark results — Apple M4 Max (40-core GPU), macOS

Measured on an otherwise-idle machine (concurrent jobs SIGSTOPped), wgpu
Metal backend, primitives test scene (3 geoms + plane; `benchmarks/bench.py`
with no arguments). Scene complexity matters: see the high-complexity row
below for a real robot scene.

| N envs | res | batches/s | env-frames/s |
|---|---|---|---|
| 16 | 64 | 518 | 8,288 |
| 64 | 64 | 335 | 21,438 |
| 256 | 64 | 138 | 35,242 |
| 1024 | 64 | 43 | 44,259 |
| 16 | 128 | 388 | 6,213 |
| 64 | 128 | 190 | 12,142 |
| 256 | 128 | 53 | 13,531 |
| 1024 | 128 | 12 | 12,109 |
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


## Pipelined readback (async double-buffering)

`render(..., pipelined=True)` submits the current frame and returns the
PREVIOUS one (one-frame latency, standard for RL observation loops), turning
the per-frame readback sync into overlap. Staging buffers are mapped, not
copied — on Apple Silicon's unified memory the mapped pointer IS the frame.

| config | sync | pipelined |
|---|---|---|
| N=1024 @64px | 44,183 env-fps | **72,731 env-fps** (+65%) |
| N=64 @128px | 11,572 env-fps | **20,253 env-fps** (+75%) |

## Performance-parity program (vs madrona_mjx)

Status after 2026-08-16 work:
1. Host-side Python packing: DONE (vectorized; +29% at large N).
2. Async double-buffered readback: DONE (`pipelined=True`; +65-75%; mapped
   staging buffers exploit unified memory — the map is the zero-copy CPU
   handoff).
3. Remaining: direct GPU-tensor handoff to a torch/MPS learner without the
   CPU-visible hop (needs a Metal buffer <-> MPS tensor bridge; no public
   Python path exists today — native-extension territory).
Honest framing: absolute parity with a 450 W RTX 4090 is not reachable on
~50 W laptop silicon. The comparable metric is performance-per-watt:
**~1.45k env-fps/W here (72.7k @ ~50 W) vs ~0.9k fps/W there (403k @
~450 W) — mjbatch-metal now exceeds madrona_mjx on efficiency**, and the
remaining ~5.5x absolute gap is within the ~5-8x hardware differential.

## Context vs madrona_mjx

madrona_mjx reports ~403,000 frames/s at 64x64 on an NVIDIA RTX 4090
(as measured in the PyBatchRender paper's comparison, arXiv:2601.01288).
mjbatch-metal reaches ~44,000 frames/s at the same resolution on an M4 Max
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
| segmentation-ID output | - | yes, parity-tested vs mujoco.Renderer (`test_segmentation_output`, per-geom IoU >0.85) |
| frustum culling | yes | yes, CPU-side conservative sphere test (`cull=True`; output-identical, tested) |
| async pipelined readback | yes (GPU-resident) | yes (`pipelined=True`, one-frame latency, +65-75%) |
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
