# CLAUDE.md — e8-eea (E8_EEA_v5)

Architectural invariants that are not obvious from reading the code, and constraints that must hold for the emotional intelligence substrate to remain grounded.

## Lyapunov hard veto — λ₁ < 0 is a hard gate, not a soft penalty

`H_meta` checks the leading Lyapunov exponent λ₁ before accepting any structural update. λ₁ < 0 is REQUIRED. A candidate that fails the veto is rejected entirely — not merged at lower weight, not averaged in, not queued for later. Do not convert this to a regularization term or soft loss. The veto enforces the timescale separation invariant below.

## Timescale separation — hardest invariant to see, easiest to break

Emotion at cycle t cannot modify weights that governed cycle t.

The invariant: weights `W_t` govern the forward pass that produces `emotion_t`, and `W_t` is updated to `W_{t+1}` only AFTER `emotion_t` is computed. If you move the weight update before the forward pass — even as a performance optimization — you break the separation proof and the Lyapunov certificate becomes invalid.

Symptom of violation: arousal-valence oscillations that fail to converge, or λ₁ drifting toward 0 over time.

## arousal and valence are derived @properties — not stored state

`EmotionalState.arousal` and `.valence` are computed from the internal activation vector each time they are accessed. They are NOT stored as attributes. Do NOT:
- Serialize them into history buffers as ground truth
- Use them as inputs to the weight update
- Cache them across cycles

If you need emotional trajectory history, store the activation vector. Storing the derived scalars loses information and creates a latent feedback loop.

## α/β/γ/δ are overwritten by sovereign_manifold every cycle

`RelationalE8Bridge.apply_to_e8_agent()` in `sovereign_manifold.py` writes `agent.alpha/beta/gamma/delta` from the relational state on every manifold cycle. E8's own internal weight updates happen on a slower clock and are clobbered by design. Relational state is higher-authority than emotional state in this architecture. If you want E8 to have autonomous weight control, you need a new negotiation layer between the two systems — not a change here.

## H_meta rejection signals frustration — consumed by sovereign_manifold

When H_meta rejects a candidate (λ₁ ≥ 0), it appends to `H_meta.history` with `accepted=False`. `sovereign_manifold.py` reads these rejection events in Phase 6 and feeds them to `FrustrationSignatureDetector`. If you change the schema of `H_meta.history` entries, update the consumer in `sovereign_manifold.py` accordingly.

## dissociation term — the min(0, valence) gate must not be removed

The dissociation term in the activation update is gated by `min(0, valence)`. This ensures the dissociative correction only fires when valence is negative (distress). Removing the gate makes the system dissociate during positive emotional states, which produces pathological activation collapse.

## Three E8 strategies map to DRA modes in sovereign_manifold

sovereign_manifold sets E8 strategy biases via `DRA.e8_strategy_bias()`:
- GENERATOR mode → strategy_3 dominant (0.70): exploratory
- WATCHER mode → strategy_2 dominant (0.55): stabilizing
- STANDARD mode → balanced (0.33/0.33/0.34)

If you add a fourth strategy, add a corresponding bias entry in `sovereign_manifold.py`.

## Input encoding — relational state, not raw sensory data

When used via sovereign_manifold, the E8 input vector comes from `RelationalE8Bridge.encode_relational_as_e8_input(s)`, which packs the 15-node relational state into a 16D vector. The network was not designed for raw sensory input in this integration. Feeding it different data changes the weight semantics without changing the architecture.
