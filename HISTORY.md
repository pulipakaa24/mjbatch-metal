# Development history

Engineering log for mjbatch-metal — the measurements, bugs, and decisions
behind the current state. The README/BENCHMARKS/ROADMAP describe what the
tool is *now*; this file records how it got there. Useful for auditing
claims, understanding trade-offs, and avoiding re-litigating settled
decisions.

All work below: 2026-08-16, Apple M4 Max (40-core GPU), macOS.

## Architecture evolution

1. **v0 — single-view wgpu renderer.** Unindexed triangle soup, flat
   per-face normals, Lambert shading, MuJoCo camera model. Validated
   side-by-side against `mujoco.Renderer` (pose/silhouette match). ~380 fps
   at 256px vs ~71 fps/process for `mujoco.Renderer`.
2. **v1 — tiled batch.** One instanced draw of the whole vertex buffer for
   all envs; per-env storage buffers; in-shader background compositing;
   fragment-shader tile clipping. 16 envs: ~1,950 env-fps. Scaling
   plateaued ~2.5k env-fps — vertex-bound: the SO-101 visual meshes total
   1.2M expanded vertices. Quadric decimation (≤1,500 faces/mesh) lifted
   the plateau to ~6k env-fps.
3. **v2 — indexed unique-mesh instancing** (the Madrona/PyBatchRender
   geometry structure): each distinct mesh stored once (indexed, smooth
   normals, UVs), one instanced draw per unique mesh via a global
   (env, slot) instance table. The 348k-face arm scene dropped from 1.2M
   expanded to 162k indexed vertices: 8.4k env-fps sync WITHOUT decimation
   (3.3x over v1 on the same scene). Cost: ~25% peak throughput on trivial
   scenes (34.4k vs 46k env-fps @64px/1024) — later recovered (see
   optimizations).
4. **Textures.** Atlas packing of MuJoCo textures, mesh UVs via
   `mesh_facetexcoord`, planar UVs for planes/boxes. Two bugs found by the
   parity test: MuJoCo texture rows are bottom-up relative to sampling
   (checker pattern rendered phase-inverted, correlation -0.88 -> fixed by
   flipping rows at atlas build), and boxes initially had no UVs (sampled a
   single texel). After fixes: pattern correlation 0.994 vs
   `mujoco.Renderer`.
5. **Feature pass:** metric depth output (readback of the depth attachment,
   perspective-linearized; median <1cm vs reference), segmentation IDs
   (r32uint MRT target; per-geom IoU >0.85 vs reference), CPU
   conservative-sphere frustum culling (output-identical; found a missing
   COPY_DST usage flag on the instance-table buffer on first run), and
   async double-buffered readback (`pipelined=True`).

## Optimization history (all verified-paused measurements)

- **Per-frame Python packing** profiled at 6.4 ms/frame (21% of frame time
  at N=1024/64px): per-env camera-matrix loop. Vectorized (batch
  quat->mat, einsum view-proj, vectorized tile rects): +29% at large N,
  shrinking the indexed-path trivial-scene penalty from ~25% to ~4%.
- **Pipelined readback**: the 16 MB atlas readback was a hard per-frame
  sync. Double-buffered staging with mapped readback (unified memory makes
  the map a zero-copy CPU handoff): +65% at N=1024/64px (44.2k -> 72.7k
  env-fps), +75% at N=64/128px (11.6k -> 20.3k).
- **Metal texture-array layer cap** (2048) broke N>2048; backgrounds now
  cycle (env e uses layer e % 2048). N=4096 verified.

## Benchmark-integrity episodes

- Early same-night benchmarks were cross-contaminated by a concurrently
  running training job; caught in review, re-measured with the job
  SIGSTOPped and process state verified (T) during every subsequent
  measurement window. A first pause attempt failed silently on a zsh
  word-splitting quirk (`kill` received one concatenated string) — later
  pauses verified via `ps` state, not exit status alone.
- A performance-per-watt superiority claim vs madrona_mjx was published
  and then withdrawn after a provenance audit: the widely-cited
  "madrona_mjx 403k fps @64px on a 4090" figure (via the PyBatchRender
  paper's comparison table) traces to the MuJoCo Playground paper, where
  it is an END-TO-END figure (MJX GPU physics + rendering) on
  CartpoleBalance, on datacenter hardware — and neither side's power draw
  had been measured. Replaced with methodology-matched end-to-end
  benchmarks (`benchmarks/end_to_end.py`) and reproduce-it-yourself
  instructions.
- The original end-to-end cartpole benchmark scene had a camera-authoring
  bug (wrong quaternion tilt; the camera missed the scene, so frames were
  empty). Caught while generating the README gallery — a reminder to look
  at renders, not just time them. Re-measured on the corrected scene:
  52.6-53.1k steps/s (within noise of the empty-frustum numbers; the
  workload is physics-thread-bound at that scale).
- The example VecEnv shipped briefly with a stale renderer call signature
  and a mangled header from scripted editing; caught in a claim-audit
  pass, fixed, and smoke-tested. Same pass verified `pip install -e` in a
  clean venv (16/16 tests from PyPI-resolved dependencies).

## Decisions

- **Dual geometry paths (monolithic vs indexed) rejected**: after
  vectorization the indexed path costs ~4% on trivial scenes and wins 3.3x
  on real ones; a second path would double the parity-tested surface.
  Revisit trigger recorded in ROADMAP.
- **LOD deferred**: no multi-resolution asset set exists to validate
  against; untested LOD would be feature theater.
- **Shadows non-goal**: discarded by the composited observation style;
  revisit only for non-composited use cases.
- **Cross-tile determinism contract**: identical calls are byte-identical;
  identical content in different atlas tiles can differ at z-fighting
  pixels of interpenetrating geometry (~0.01% measured on a stress scene)
  because per-tile NDC offsets shift rasterization sub-pixel.
