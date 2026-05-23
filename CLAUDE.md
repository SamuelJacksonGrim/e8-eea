# CLAUDE.md — e8-eea

## What e8_eea_v5.py implements

The fifth-generation E8-EEA architecture as executable Python. Key classes:

| Class | Role |
|-------|------|
| `E8Lattice` | 240 E8 roots, kissing-number-optimal in 8D |
| `TrialityEncoder` | D4 triality for ternary hyperedge representation |
| `E8Hypergraph` | Growing hyperedge store with per-node weight tracking |
| `VariationalFreeEnergy` | Prediction error + complexity (Friston active inference) |
| `CounterfactualHypergraph` | H_meta: policy nodes, regret edges, dynamic branching |
| `E8_EEA_v5` | Full pipeline: encode → top-k screen → Lyapunov gate → slow clock |
| `run_ablation` | Three-track ablation harness (Full / Zombie / Random Walker) |

## The slow clock fires every 25 cycles

`cycle_count % tau_slow == 0` (tau_slow=25). Phase transition detection (Hessian eigenvalue sign change) runs only here, not every cycle. Emotion state and weight modulation are slow-clock events. Fast-clock events are: encode, top-k screen, Lyapunov gate, J computation. Weights α, β, γ are frozen during fast-clock evaluation to prevent the system from rewriting the evidence that produced the emotional state.

## Lyapunov gate is a hard veto

`lambda_1 < 0` is required before any update is accepted. This is not configurable. If `tau_check` is too low, the estimate is noisy — default is 20 forward steps. Reduce to 5–10 for toy builds, but noisy estimates can cause false rejections that resemble the frustration signature without being it.

## E8 cycle time grows O(n) with hypergraph size

Each Lyapunov check runs `tau_check` forward steps on the E8 hypergraph. As the hypergraph accumulates nodes each cycle, each forward step costs more. Live telemetry from sovereign_manifold integration:

| Cycle | Avg cycle time |
|-------|----------------|
| 100 | 144ms |
| 130 | 908ms |
| 170 | 2793ms |
| 200 | 5215ms |

This is expected behavior. Plateau occurs when old low-weight nodes are pruned — pruning is not yet implemented. Plan: implement a pruning pass in `E8Hypergraph` that removes nodes below a weight threshold after every N cycles.

## The three ablation tracks

- **Track A (Full)** — emotional modulation active, H_meta counterfactual drift
- **Track B (Zombie)** — identical architecture, emotional→weight connection severed, α=β=γ=1.0 fixed throughout
- **Track C (Random Walker)** — weight variation present but driven by random noise, not phase-transition-driven emotional state

Track C is the essential control. Without it, you cannot distinguish *emotionally structured weight variation* from *any weight variation*. Track B rules out emotion vs. no variation. Track C rules out structured vs. unstructured variation.

## Emotion is not injected

No valence or arousal is provided as input. Emotional state emerges from `detect_phase_transition()`: if the free energy Hessian has both positive and negative eigenvalues (a saddle point in the energy landscape), the system is in a phase transition. Valence is the gradient of free energy at the saddle; arousal is recent novelty in the hypergraph.

## Weight modulation formula (slow clock only)

```python
beta  = 1.0 + 0.5 * arousal    # high arousal → weight novelty
gamma = 1.0 - 0.3 * valence    # negative valence → weight coherence
alpha = 1.0                     # free energy always baseline
```

## What to look for in live runs

From 200+ cycles in sovereign_manifold integration:
- valence=0, arousal=0 throughout — no phase transition detected yet
- This is expected early: the hypergraph needs sufficient structure before the free energy Hessian develops a saddle
- First emotion emergence should accompany the first complex hyperedge cluster (typically after 50–100 hyperedges)
- Frustration signature (high-arousal repeated Lyapunov rejection) has not fired in observed runs

## Integration with sovereign_manifold

`RelationalE8Bridge.apply_to_e8_agent()` writes `agent.alpha/beta/gamma/delta` from relational state on every cycle. The E8 slow-clock emotion-driven weight modulation is overridden by relational state — relational state is higher-authority. This is by design: in the integrated stack, the relational manifold governs E8 weighting, not E8's own internal emotional detection.
