# Roadmap

Scope policy: features enter when a measurement or a user demands them, and
every feature ships with a parity test against `mujoco.Renderer`. Current
scope covers the domain-randomized, background-composited observation style
(realism from composited photos, not rendered detail).

## Near-term (feasible additions, ordered by expected demand)

1. **Textures.** MuJoCo provides `mesh_texcoord` and material→texture
   bindings; implementation is a texture atlas + UV vertex attribute +
   fragment sampling (~days including the parity tests for filtering,
   color space, and UV conventions — the tests are the real work).
   Closes the main fidelity gap vs Madrona's textured-scene support.
2. **Segmentation-ID output.** The instanced geom ID is already in the
   vertex stream; exposing it as an optional second render target is small
   and enables mask-based pipelines that don't want compositing.
3. **Non-square tiles / per-env resolution presets; camera intrinsics from
   `sensorsize`/`focal`** (currently fovy-based projection only).

## Larger rework (waits for a concrete use case)

4. **High-complexity scenes** (Habitat-class, millions of triangles):
   requires indexed geometry, one instanced draw per unique mesh (instead
   of one giant unindexed buffer), frustum culling, and probably LOD. This
   is the architecture Madrona/PyBatchRender use; ours trades it away for
   simplicity at robot-scene scale (tens of geoms).
5. **Shadows / richer shading** for users whose observations are NOT
   composited. Deliberate non-goal while compositing covers realism.

## Out of scope

- **GPU-resident physics (MJX-class).** A JAX/XLA-on-Metal problem, not a
  rendering problem. CPU MuJoCo (threaded) is not the bottleneck on Apple
  Silicon at the batch sizes this renderer serves.
- **Photorealism.** Both this project and Madrona delegate that tier to a
  second engine (here: Unreal Engine; in the NVIDIA ecosystem: Isaac
  RTX/Omniverse).
