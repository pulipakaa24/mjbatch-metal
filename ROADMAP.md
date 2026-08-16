# Roadmap

Scope policy: features enter when a measurement or a user demands them, and
every feature ships with a parity test against `mujoco.Renderer`. Current
scope covers the domain-randomized, background-composited observation style
(realism from composited photos, not rendered detail).

## Planned

1. **Non-square tiles / per-env resolution presets; camera intrinsics from
   `sensorsize`/`focal`** (currently fovy-based projection only).
2. **LOD for Habitat-class scenes** (millions of triangles) — waits for a
   real multi-resolution asset set to validate against; until then, manual
   decimation levels are available via `decimate_faces`.
3. **Direct GPU-tensor handoff** to a torch/MPS learner (Metal buffer <->
   MPS tensor bridge; native-extension territory).

## Considered, not planned

- **Shadows / richer shading**: non-goal while the composited observation
  style covers realism; revisit only for non-composited use cases.
- **Dual geometry paths (monolithic vs indexed, auto-selected)**: the
  indexed path costs ~4% peak on trivial scenes and wins 3.3x on real ones;
  a second path would double the parity-tested surface for negligible gain.
  Revisit only if a real workload puts rendering on the critical path AND
  profiling shows a monolithic path winning >=2x for it.

## Out of scope

- **GPU-resident physics (MJX-class).** A JAX/XLA-on-Metal problem, not a
  rendering problem. CPU MuJoCo (threaded) is not the bottleneck on Apple
  Silicon at the batch sizes this renderer serves.
- **Photorealism.** Both this project and Madrona delegate that tier to a
  second engine (here: Unreal Engine; in the NVIDIA ecosystem: Isaac
  RTX/Omniverse).
