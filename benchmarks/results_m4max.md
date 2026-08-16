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

### End-to-end comparison vs MuJoCo Playground (MJX + madrona_mjx)

MuJoCo Playground's published figures are END-TO-END environment steps/s
(GPU physics + rendering, no learner). `benchmarks/end_to_end.py` measures
the same quantity on this stack (threaded CPU MuJoCo physics + batched
pipelined rendering). Same methodology, different hardware — disclosed:

| scene class | theirs (datacenter GPU, MJX physics) | ours (M4 Max laptop, CPU physics) |
|---|---|---|
| cartpole-class, 64px | ~403,000 steps/s (CartpoleBalance) | **52,600-53,100 steps/s** (N=1024-4096; plateau = CPU physics-thread saturation) |
| arm pick-cube class, 128px | ~37,000 steps/s (PandaPickCubeCartesian, ~resolution-insensitive) | **16,957 steps/s** (SO-101 arm scene, N=64) |

Reading: on trivial scenes their GPU-resident physics dominates (7.5x);
on robot-scale scenes — the ones people train — the gap is **2.2x**, on a
laptop, with no CUDA. Neither side's power draw is measured; no perf/watt
claim is made (a 4090/A100-class board alone draws several times this
laptop's total power).

Note the provenance of the widely-cited "403k fps" number: it is this
end-to-end CartpoleBalance figure from the MuJoCo Playground paper, not a
renderer-only benchmark; render-only comparisons against it are
apples-to-oranges in both directions.

### Gallery

Rendered examples (regenerate with `python benchmarks/make_gallery.py
[robot_scene.xml]`): tiled batch atlas, DR/compositing grid, RGB+depth+seg
triptych, and the texture-parity side-by-side live in `assets/` and are
embedded in the README. A camera-authoring bug in the original cartpole
benchmark scene (camera missed the scene entirely) was caught by LOOKING at
these renders — the end-to-end numbers above are from the corrected scene.

### Reproduce it yourself

```
python benchmarks/bench.py                    # render-only grid
python benchmarks/bench.py your_scene.xml     # your own scene
python benchmarks/end_to_end.py               # physics+render, cartpole-class
python benchmarks/end_to_end.py scene.xml 64 128   # your robot scene
# real power draw during a run (macOS, needs sudo):
sudo powermetrics --samplers gpu_power -i 500
```
If you have access to an NVIDIA machine with madrona_mjx, the honest
comparison is: same scene translated to both, same resolution, same batch,
render-only timings isolated on both sides, wall power measured on both
sides. We would welcome such a result as an issue/PR.

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
| frustum culling | not verified from their docs | yes, CPU-side conservative sphere test (`cull=True`; output-identical, tested) |
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
