# Roadmap

Scope policy: features enter when a measurement or a user demands them, and
every feature ships with a parity test against `mujoco.Renderer`. Current
scope covers the domain-randomized, background-composited observation style
(realism from composited photos, not rendered detail).

## Near-term (feasible additions, ordered by expected demand)

1. ~~**Textures.**~~ DONE (2026-08-16): atlas-packed MuJoCo textures with
   mesh/plane/box UVs, parity-tested (pattern correlation 0.994).
2. **Segmentation-ID output.** The instanced geom ID is already in the
   vertex stream; exposing it as an optional second render target is small
   and enables mask-based pipelines that don't want compositing.
3. **Non-square tiles / per-env resolution presets; camera intrinsics from
   `sensorsize`/`focal`** (currently fovy-based projection only).

## Larger rework (waits for a concrete use case)

4. **High-complexity scenes**: indexed geometry + per-unique-mesh
   instanced draws DONE (2026-08-16; 348k-tri scene at 8.4K env-fps without
   decimation). Remaining for Habitat-class (millions of triangles):
   frustum culling and LOD.
5. **Shadows / richer shading** for users whose observations are NOT
   composited. Deliberate non-goal while compositing covers realism.

## Out of scope

- **GPU-resident physics (MJX-class).** A JAX/XLA-on-Metal problem, not a
  rendering problem. CPU MuJoCo (threaded) is not the bottleneck on Apple
  Silicon at the batch sizes this renderer serves.
- **Photorealism.** Both this project and Madrona delegate that tier to a
  second engine (here: Unreal Engine; in the NVIDIA ecosystem: Isaac
  RTX/Omniverse).
