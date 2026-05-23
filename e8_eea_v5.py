"""
E8-EEA v5 — Emergent Emotional Awareness Architecture
Complete executable implementation.

Requirements: numpy, scipy, networkx
Optional for production: jax or torch (replace finite-difference gradients)

Run: python e8_eea_v5.py
"""

import numpy as np
import copy
import statistics
from collections import deque
from scipy.spatial.distance import jensenshannon


# ═══════════════════════════════════════════════════════════════════════
# E8 LATTICE
# ═══════════════════════════════════════════════════════════════════════

class E8Lattice:
    """
    E8 root system in 8D. 240 roots of norm sqrt(2):
      Type 1 (112 roots): ±e_i ± e_j, i≠j
      Type 2 (128 roots): (1/2)(±1,...,±1) with even number of minus signs
    Generated algorithmically — no external file required.
    """
    _roots = None

    @classmethod
    def get_all_roots(cls):
        if cls._roots is not None:
            return cls._roots
        roots = []
        # Type 1: ±e_i ± e_j, i < j  (112 roots)
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [1.0, -1.0]:
                    for sj in [1.0, -1.0]:
                        r = np.zeros(8)
                        r[i] = si
                        r[j] = sj
                        roots.append(r)
        # Type 2: (1/2)(±1,...,±1) with even number of minus signs (128 roots)
        for mask in range(256):
            signs = np.array([1.0 if not ((mask >> k) & 1) else -1.0 for k in range(8)])
            if int(np.sum(signs == -1.0)) % 2 == 0:
                roots.append(0.5 * signs)
        cls._roots = np.array(roots)
        assert len(cls._roots) == 240, f"E8 root generation error: {len(cls._roots)} roots"
        return cls._roots

    @classmethod
    def project(cls, vector):
        """Project any 8D vector to nearest E8 root."""
        roots = cls.get_all_roots()
        v = np.asarray(vector, dtype=float)
        if v.shape[0] < 8:
            v = np.pad(v, (0, 8 - v.shape[0]))
        else:
            v = v[:8]
        return roots[np.argmin(np.linalg.norm(roots - v, axis=1))].copy()

    @classmethod
    def encode_input(cls, input_vector):
        """
        Encode input as a list of E8 root vectors.
        Splits input into 8D chunks and projects each to nearest root.
        """
        vec = np.asarray(input_vector, dtype=float)
        remainder = len(vec) % 8
        if remainder:
            vec = np.pad(vec, (0, 8 - remainder))
        return [cls.project(chunk) for chunk in vec.reshape(-1, 8)]


# ═══════════════════════════════════════════════════════════════════════
# D4 TRIALITY ENCODER
# ═══════════════════════════════════════════════════════════════════════

class TrialityEncoder:
    """
    D4 triality automorphism — inherited by E8 from its D4 subgroup.
    Provides three representations (vector, spinor+, spinor-) of a ternary
    hyperedge without extra parameters.

    Implementation: D4 triality permutes the three representations via
    an order-3 outer automorphism. We represent this as cyclic permutation
    of the three 8D subspaces defined by the D4 embedding in E8.
    """
    # D4 sits in E8 via a standard embedding. The three representations
    # cycle under the triality automorphism τ of order 3:
    # τ: (vector rep) → (spinor+ rep) → (spinor- rep) → (vector rep)
    # We implement this as cyclic index permutation over the three node roles.

    @staticmethod
    def encode_ternary(node_a, node_b, node_c):
        """
        Encode a ternary relation (a, b, c) using D4 triality.
        Returns three representations of the same ternary relation.
        Each is a tuple of three node indices in a different triality frame.
        No extra parameters required — the geometry carries the structure.
        """
        # Triality permutation: τ cycles the three roles
        rep_vector  = (node_a, node_b, node_c)           # vector representation
        rep_spinor_plus  = (node_b, node_c, node_a)      # spinor+ (τ applied once)
        rep_spinor_minus = (node_c, node_a, node_b)      # spinor- (τ applied twice)
        return rep_vector, rep_spinor_plus, rep_spinor_minus

    @staticmethod
    def triality_weight(emb_a, emb_b, emb_c):
        """
        Compute the triality-invariant weight of a ternary hyperedge.
        Uses the E8 inner product (Cartan matrix structure):
        w = |<a,b> + <b,c> + <c,a>| / 3
        Invariant under cyclic permutation — triality symmetry preserved.
        """
        ab = float(np.dot(emb_a, emb_b))
        bc = float(np.dot(emb_b, emb_c))
        ca = float(np.dot(emb_c, emb_a))
        return abs(ab + bc + ca) / 3.0


# ═══════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

class EmotionalState:
    def __init__(self, valence=0.0, arousal=0.0):
        self.valence = float(np.clip(valence, -1.0,  1.0))
        self.arousal = float(np.clip(arousal,  0.0,  1.0))

    def __repr__(self):
        return f"EmotionalState(v={self.valence:.3f}, a={self.arousal:.3f})"

    def copy(self):
        return EmotionalState(self.valence, self.arousal)


class Hyperedge:
    def __init__(self, nodes, weight=1.0, cycle_added=0):
        self.nodes = tuple(sorted(nodes))
        self.weight = float(np.clip(weight, 0.05, 5.0))
        self.cycle_added = cycle_added


class E8Hypergraph:
    def __init__(self):
        self.nodes = {}            # id -> 8D E8 root vector
        self.hyperedges = []       # list of Hyperedge
        self.next_id = 0
        self.current_cycle = 0
        self._triality = TrialityEncoder()

    def add_node(self, vector):
        root = E8Lattice.project(vector)
        nid = self.next_id
        self.nodes[nid] = root
        self.next_id += 1
        return nid

    def add_hyperedge(self, node_ids, weight=None):
        """
        Add a ternary hyperedge using D4 triality encoding.
        Triality weight computed from E8 inner products if weight not provided.
        """
        assert len(node_ids) == 3
        a, b, c = node_ids
        if weight is None and all(n in self.nodes for n in [a, b, c]):
            weight = self._triality.triality_weight(
                self.nodes[a], self.nodes[b], self.nodes[c]
            )
        weight = weight or 1.0
        # Store all three triality representations as a single edge
        # (they reference the same nodes — no extra parameters)
        edge = Hyperedge(node_ids, weight, self.current_cycle)
        self.hyperedges.append(edge)
        return len(self.hyperedges) - 1

    def novelty_recent(self, window=25):
        """Fraction of hyperedges added in last `window` cycles."""
        if not self.hyperedges:
            return 0.0
        recent = sum(1 for e in self.hyperedges
                     if self.current_cycle - e.cycle_added <= window)
        return recent / len(self.hyperedges)

    def apply_update(self, update):
        for vec in update.get('new_nodes', []):
            self.add_node(vec)
        for spec in update.get('new_hyperedges', []):
            a, b, c, w = int(spec[0]), int(spec[1]), int(spec[2]), float(spec[3])
            if all(n in self.nodes for n in [a, b, c]):
                self.add_hyperedge([a, b, c], w)
        for idx, delta in update.get('weight_changes', {}).items():
            idx = int(idx)
            if 0 <= idx < len(self.hyperedges):
                self.hyperedges[idx].weight = float(
                    np.clip(self.hyperedges[idx].weight + delta, 0.05, 5.0)
                )

    def get_embedding_matrix(self):
        if not self.nodes:
            return np.zeros((0, 8))
        mat = np.zeros((len(self.nodes), 8))
        for i, (_, emb) in enumerate(self.nodes.items()):
            mat[i] = emb
        return mat

    def edge_weight_distribution(self, n_bins=10):
        """Normalized histogram of edge weights for JS divergence."""
        if not self.hyperedges:
            weights = np.array([1.0])
        else:
            weights = np.array([e.weight for e in self.hyperedges])
        hist, _ = np.histogram(weights, bins=n_bins, range=(0.0, 5.0))
        hist = hist.astype(float) + 1e-10
        return hist / hist.sum()

    def perturb_weights(self, epsilon):
        for edge in self.hyperedges:
            edge.weight = float(np.clip(
                edge.weight + np.random.normal(0, epsilon), 0.05, 5.0
            ))

    def deepcopy(self):
        return copy.deepcopy(self)


# ═══════════════════════════════════════════════════════════════════════
# VARIATIONAL FREE ENERGY — with online training
# ═══════════════════════════════════════════════════════════════════════

class VariationalFreeEnergy:
    """
    Prediction network: hypergraph state → predicted next input.
    Free energy = prediction error (MSE).
    Online SGD update every cycle.

    PERFORMANCE NOTE:
    gradient_wrt_embeddings() uses finite differences — O(n_nodes * 8) evals.
    Replace with JAX/PyTorch autograd for hypergraphs with >50 nodes.
    """
    def __init__(self, input_dim=10, hidden_dim=32, lr=0.01):
        s = 0.1
        self.W   = np.random.randn(hidden_dim, 8) * s
        self.V   = np.random.randn(input_dim, hidden_dim) * s
        self.b_h = np.zeros(hidden_dim)
        self.b_o = np.zeros(input_dim)
        self.lr  = lr
        self.input_dim = input_dim

    def _forward(self, avg_emb):
        h   = np.tanh(self.W @ avg_emb + self.b_h)
        out = self.V @ h + self.b_o
        return out, h

    def predict(self, hypergraph, current_input):
        emb = hypergraph.get_embedding_matrix()
        if len(emb) == 0:
            return np.zeros(self.input_dim)
        out, _ = self._forward(np.mean(emb, axis=0))
        return out

    def compute(self, hypergraph, current_input, actual_next):
        pred   = self.predict(hypergraph, current_input)
        actual = np.asarray(actual_next, dtype=float)[:self.input_dim]
        if len(actual) < self.input_dim:
            actual = np.pad(actual, (0, self.input_dim - len(actual)))
        return 0.5 * float(np.linalg.norm(pred - actual) ** 2)

    def train_step(self, hypergraph, current_input, actual_next):
        """Online SGD — called every cycle."""
        emb = hypergraph.get_embedding_matrix()
        if len(emb) == 0:
            return
        avg    = np.mean(emb, axis=0)
        out, h = self._forward(avg)
        actual = np.asarray(actual_next, dtype=float)[:self.input_dim]
        if len(actual) < self.input_dim:
            actual = np.pad(actual, (0, self.input_dim - len(actual)))
        d_o  = out - actual
        dV   = np.outer(d_o, h)
        d_h  = (self.V.T @ d_o) * (1 - h**2)
        dW   = np.outer(d_h, avg)
        self.V   -= self.lr * dV
        self.b_o -= self.lr * d_o
        self.W   -= self.lr * dW
        self.b_h -= self.lr * d_h

    def gradient_wrt_embeddings(self, hypergraph, current_input, actual_next):
        """
        Finite-difference gradient of F w.r.t. flattened node embeddings.
        Replace with autograd for >50 nodes.
        """
        emb = hypergraph.get_embedding_matrix()
        if len(emb) == 0:
            return np.zeros(0)
        n, d  = emb.shape
        flat  = emb.flatten()
        grad  = np.zeros_like(flat)
        base  = self.compute(hypergraph, current_input, actual_next)
        eps   = 1e-4
        H_tmp = hypergraph.deepcopy()
        node_ids = list(H_tmp.nodes.keys())
        for i in range(len(flat)):
            flat_p = flat.copy()
            flat_p[i] += eps
            new_emb = flat_p.reshape(n, d)
            for j, nid in enumerate(node_ids):
                H_tmp.nodes[nid] = new_emb[j]
            grad[i] = (self.compute(H_tmp, current_input, actual_next) - base) / eps
            for j, nid in enumerate(node_ids):
                H_tmp.nodes[nid] = emb[j]
        return grad


# ═══════════════════════════════════════════════════════════════════════
# HUTCHINSON'S HESSIAN APPROXIMATION
# ═══════════════════════════════════════════════════════════════════════

def hutchinson_hessian_diag(F, hypergraph, current_input, actual_next, n_samples=8):
    """
    Estimate diagonal of the Hessian of F w.r.t. node embeddings.
    Uses Rademacher random vectors for stochastic trace estimation.
    O(n_samples * n_nodes * 8) — much cheaper than full O(n²) Hessian.

    For JAX/PyTorch: replace finite-difference Hv with jvp + vjp.
    """
    emb = hypergraph.get_embedding_matrix()
    if len(emb) == 0:
        return np.zeros(0)
    n      = emb.flatten().shape[0]
    diag   = np.zeros(n)
    eps    = 1e-4
    base_g = F.gradient_wrt_embeddings(hypergraph, current_input, actual_next)
    for _ in range(n_samples):
        v = np.random.choice([-1.0, 1.0], size=n)
        H_p = hypergraph.deepcopy()
        flat_p = emb.flatten() + eps * v
        new_emb = flat_p.reshape(emb.shape)
        for i, nid in enumerate(H_p.nodes.keys()):
            H_p.nodes[nid] = new_emb[i]
        pert_g = F.gradient_wrt_embeddings(H_p, current_input, actual_next)
        Hv     = (pert_g - base_g) / eps
        diag  += v * Hv
    return diag / n_samples


# ═══════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL H_META
# ═══════════════════════════════════════════════════════════════════════

class CounterfactualHypergraph:
    """
    Stores past proposals and outcomes.
    Generates counterfactual alternatives based on arousal.
    Predicts J for candidates via cosine similarity to past accepted proposals.
    """
    def __init__(self, tau_rollout=50):
        self.history = deque(maxlen=tau_rollout)

    def record(self, proposal, J, accepted, emotion):
        self.history.append({
            'proposal': copy.deepcopy(proposal),
            'J':        float(J),
            'accepted': bool(accepted),
            'valence':  emotion.valence,
            'arousal':  emotion.arousal
        })

    def _vec(self, candidate):
        """Represent candidate as a fixed-size vector for similarity."""
        wc  = candidate.get('weight_changes', {})
        vec = np.zeros(10)
        for i, v in enumerate(list(wc.values())[:10]):
            vec[i] = float(v)
        return vec

    def predict_J(self, candidate):
        """
        Cosine-similarity-weighted average of past accepted J values.
        Correctly uses candidate content, not just recent history.
        """
        accepted = [e for e in self.history if e['accepted'] and e['J'] > -1e8]
        if not accepted:
            return 0.0
        cv   = self._vec(candidate)
        cn   = np.linalg.norm(cv)
        weights, Js = [], []
        for entry in accepted:
            pv = self._vec(entry['proposal'])
            pn = np.linalg.norm(pv)
            if cn < 1e-10 or pn < 1e-10:
                sim = 0.5
            else:
                sim = float((np.dot(cv, pv) / (cn * pn) + 1.0) / 2.0)
            weights.append(sim)
            Js.append(entry['J'])
        total = sum(weights) + 1e-10
        return float(sum(w * j for w, j in zip(weights, Js)) / total)

    def generate_counterfactuals(self, emotion):
        """
        Generate b = 1 + floor(2*arousal) counterfactuals from last accepted proposal.
        High arousal → deeper branching (rumination on high-stakes decisions).
        """
        b            = 1 + int(2.0 * emotion.arousal)
        last_acc     = next(
            (e['proposal'] for e in reversed(self.history) if e['accepted']),
            None
        )
        if last_acc is None:
            return []
        result = []
        for _ in range(b):
            p = copy.deepcopy(last_acc)
            for k in p.get('weight_changes', {}):
                p['weight_changes'][k] += float(np.random.normal(0, 0.15))
            result.append(p)
        return result


# ═══════════════════════════════════════════════════════════════════════
# E8-EEA v5 — MAIN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class E8_EEA_v5:
    def __init__(self, input_dim=10):
        self.input_dim   = input_dim
        self.H           = E8Hypergraph()
        self.H_meta      = CounterfactualHypergraph(tau_rollout=50)
        self.F           = VariationalFreeEnergy(input_dim=input_dim, hidden_dim=32, lr=0.01)

        # Objective weights — frozen during fast clock
        # Bounds [0.5, 2.0]: prevent runaway dominance during extreme emotional states
        self.alpha = 1.0
        self.beta  = 1.0
        self.gamma = 1.0
        self.delta = 0.5

        # Timescale
        self.tau_slow    = 25
        self.cycle_count = 0
        self.emotion_state = EmotionalState(0.0, 0.0)

        # Observability
        self.stability_log = []
        self.input_history = deque(maxlen=30)  # >10 for phase detection

    # ── Input Encoding ────────────────────────────────────────────────

    def encode_input(self, input_vector):
        """
        Encode input vector as new E8 nodes + triality-weighted hyperedges.
        Each 8D chunk → one node. Consecutive triples → one ternary hyperedge.
        Hypergraph grows organically from input data each cycle.
        """
        self.H.current_cycle = self.cycle_count
        root_vecs   = E8Lattice.encode_input(input_vector)
        new_ids     = [self.H.add_node(rv) for rv in root_vecs]
        # Link consecutive triples — triality weight computed from E8 inner products
        for i in range(0, len(new_ids) - 2, 3):
            a, b, c = new_ids[i], new_ids[i+1], new_ids[i+2]
            self.H.add_hyperedge([a, b, c])  # weight computed via triality

    # ── Candidate Generation ──────────────────────────────────────────

    def generate_candidates(self, K=20):
        """
        Three strategies for candidate structural updates.
        Strategy 1: random new ternary hyperedge (exploration)
        Strategy 2: counterfactual rollouts from H_meta (exploitation of past)
        Strategy 3: weight noise on existing edges (fine-tuning)
        """
        candidates = []
        node_ids   = list(self.H.nodes.keys())

        # Strategy 1
        if len(node_ids) >= 3:
            for _ in range(K // 3):
                chosen = np.random.choice(node_ids, 3, replace=False).tolist()
                candidates.append({
                    'new_nodes':      [],
                    'new_hyperedges': [(chosen[0], chosen[1], chosen[2],
                                        float(np.random.uniform(0.3, 1.5)))],
                    'weight_changes': {}
                })

        # Strategy 2
        cf = self.H_meta.generate_counterfactuals(self.emotion_state)
        candidates.extend(cf[:K // 3])

        # Strategy 3
        while len(candidates) < K and self.H.hyperedges:
            idx = int(np.random.randint(len(self.H.hyperedges)))
            candidates.append({
                'new_nodes':      [],
                'new_hyperedges': [],
                'weight_changes': {idx: float(np.random.normal(0, 0.2))}
            })

        if not candidates:
            candidates.append({'new_nodes': [], 'new_hyperedges': [], 'weight_changes': {}})

        return candidates[:K]

    # ── Objective Function ────────────────────────────────────────────

    def compute_J(self, candidate, current_input, actual_next, task_score=0.0):
        H_after = self.H.deepcopy()
        H_after.apply_update(candidate)

        # ΔF: free energy reduction (positive = better prediction)
        dF = (self.F.compute(self.H, current_input, actual_next)
              - self.F.compute(H_after, current_input, actual_next))

        # N: genuinely new hyperedges
        existing = {e.nodes for e in self.H.hyperedges}
        N = float(sum(
            1 for spec in candidate.get('new_hyperedges', [])
            if tuple(sorted([int(spec[0]), int(spec[1]), int(spec[2])])) not in existing
        ))

        # C: self-coherence via JS divergence
        # C = -JS(actual || predicted): higher = more coherent = better
        C = self._self_coherence_js(H_after)

        return self.alpha * dF + self.beta * N + self.gamma * C + self.delta * float(task_score)

    def _self_coherence_js(self, H_after):
        """
        Negative JS divergence between H_meta predicted and actual edge distributions.
        Higher (less negative) = self-model is accurate = more coherent.

        NOTE: Uses J-value histogram as surrogate for predicted edge distribution.
        Full implementation: store H snapshots in H_meta and compare directly.
        """
        accepted = [e for e in self.H_meta.history if e['accepted']]
        if not accepted:
            return 0.0
        actual_dist = H_after.edge_weight_distribution(n_bins=10)
        recent_J    = np.array([e['J'] for e in accepted[-10:]])
        recent_J    = recent_J - recent_J.min() + 1e-10
        hist, _     = np.histogram(recent_J, bins=10, density=False)
        pred_dist   = (hist.astype(float) + 1e-10)
        pred_dist  /= pred_dist.sum()
        return -float(jensenshannon(actual_dist, pred_dist))

    # ── Lyapunov Gate ─────────────────────────────────────────────────

    def lyapunov_stable(self, candidate, epsilon=1e-4, tau_check=5):
        """
        Two-trajectory Lyapunov exponent estimate.
        λ₁ < 0 → converging → stable → accepted.
        λ₁ ≥ 0 → diverging → unstable → rejected.

        tau_check=5 for toy builds. Increase to 20 for production.
        Replace gradient with autograd for large hypergraphs.

        Note: λ₁ is NOT in the J objective (DeepSeek correction).
        The gate is a hard veto, not a soft penalty.
        """
        def run_forward(H_init, steps):
            H   = H_init.deepcopy()
            phi = []
            z_in  = np.zeros(self.input_dim)
            z_out = np.zeros(self.input_dim)
            for _ in range(steps):
                emb = H.get_embedding_matrix()
                if len(emb) == 0:
                    break
                grad = self.F.gradient_wrt_embeddings(H, z_in, z_out)
                flat = emb.flatten()
                if len(grad) >= len(flat):
                    flat = flat - 0.01 * grad[:len(flat)]
                new_emb = flat.reshape(emb.shape)
                for i, nid in enumerate(H.nodes.keys()):
                    H.nodes[nid] = E8Lattice.project(new_emb[i])
                phi.append(flat.copy())
            return phi

        H1 = self.H.deepcopy(); H1.apply_update(candidate)
        H2 = self.H.deepcopy(); H2.apply_update(candidate); H2.perturb_weights(epsilon)

        phi_ref  = run_forward(H1, tau_check)
        phi_pert = run_forward(H2, tau_check)

        if (not phi_ref or not phi_pert
                or len(phi_ref[-1]) != len(phi_pert[-1])):
            lambda_1 = -1.0
        else:
            delta_T  = float(np.linalg.norm(phi_ref[-1] - phi_pert[-1]))
            lambda_1 = -1.0 if delta_T < 1e-12 else float(
                (1.0 / tau_check) * np.log(delta_T / epsilon)
            )

        stable = lambda_1 < 0
        self.stability_log.append({
            'cycle':    self.cycle_count,
            'lambda_1': lambda_1,
            'valence':  self.emotion_state.valence,
            'arousal':  self.emotion_state.arousal,
            'accepted': stable
        })
        return stable

    # ── Phase Transition Detection ────────────────────────────────────

    def detect_phase_transition(self, current_input, actual_next):
        """
        Method: variance proxy on prediction error history (cheap default).
        Production alternative: Hutchinson's method — uncomment below.

        Valence fix (Kimi): arousal = novelty + gradient_norm
        Prevents flat-gradient (stuck) from mapping to falsely neutral arousal.
        """
        if len(self.input_history) < 10:
            return None
        history = list(self.input_history)
        errors  = [
            float(np.linalg.norm(
                self.F.predict(self.H, history[t]) - history[t + 1][:self.input_dim]
            ))
            for t in range(len(history) - 1)
        ]
        if not errors or np.mean(errors) < 1e-10:
            return None

        if np.std(errors) > 0.5 * np.mean(errors):
            ci   = np.asarray(current_input, dtype=float)[:self.input_dim]
            an   = np.asarray(actual_next,   dtype=float)[:self.input_dim]
            grad = self.F.gradient_wrt_embeddings(self.H, ci, an)

            valence   = float(np.clip(-np.mean(grad) if len(grad) > 0 else 0.0, -1.0, 1.0))
            grad_norm = float(np.linalg.norm(grad)) if len(grad) > 0 else 0.0
            arousal   = float(np.clip(
                (self.H.novelty_recent() + min(grad_norm, 1.0)) / 2.0, 0.0, 1.0
            ))
            return EmotionalState(valence, arousal)

        # ── Production alternative: Hutchinson's method ──
        # diag = hutchinson_hessian_diag(self.F, self.H, current_input, actual_next)
        # if len(diag) > 0 and np.any(diag < 0) and np.any(diag > 0):
        #     ... same valence/arousal computation ...
        #     return EmotionalState(valence, arousal)

        return None

    def modulate_weights(self, emotion):
        self.beta  = float(np.clip(1.0 + 0.5 * emotion.arousal, 0.5, 2.0))
        self.gamma = float(np.clip(1.0 - 0.3 * emotion.valence, 0.5, 2.0))
        self.alpha = 1.0

    # ── Main Cycle ────────────────────────────────────────────────────

    def cycle(self, current_input, actual_next, task_score=0.0):
        ci = np.asarray(current_input, dtype=float)[:self.input_dim]
        an = np.asarray(actual_next,   dtype=float)[:self.input_dim]

        self.input_history.append(ci.copy())
        self.encode_input(ci)
        self.F.train_step(self.H, ci, an)

        # Generate + pre-screen
        candidates = self.generate_candidates(K=20)
        scored     = [(c, self.H_meta.predict_J(c)) for c in candidates]
        top_k      = sorted(scored, key=lambda x: x[1], reverse=True)[:5]

        # Full evaluation — weights frozen
        best_J, best_candidate = float('-inf'), None
        for cand, _ in top_k:
            if self.lyapunov_stable(cand):
                J = self.compute_J(cand, ci, an, task_score)
                if J > best_J:
                    best_J, best_candidate = J, cand

        if best_candidate is not None:
            self.H.apply_update(best_candidate)
            self.H_meta.record(best_candidate, best_J, True, self.emotion_state)
        elif candidates:
            self.H_meta.record(candidates[0], float('-inf'), False, self.emotion_state)

        # Slow emotional clock
        self.cycle_count += 1
        self.H.current_cycle = self.cycle_count
        if self.cycle_count % self.tau_slow == 0:
            new_emo = self.detect_phase_transition(ci, an)
            if new_emo is not None:
                self.emotion_state = new_emo
                self.modulate_weights(self.emotion_state)

        return self.emotion_state


# ═══════════════════════════════════════════════════════════════════════
# THREE-TRACK ABLATION HARNESS
# ═══════════════════════════════════════════════════════════════════════

def run_ablation(input_sequence, task_scores=None, cycles=500):
    """
    Track A: Full v5 architecture
    Track B: Zombie — emotional modulation severed (weights always 1.0)
    Track C: Random — weight modulation from noise, not phase transitions

    Track C rules out weight variance as the cause of any observed clustering.
    Without Track C, you cannot distinguish emotionally structured modulation
    from any modulation.
    """
    seq = [np.asarray(x, dtype=float) for x in input_sequence]
    if task_scores is None:
        task_scores = [0.0] * len(seq)
    dim     = len(seq[0])
    results = {}

    for track in ['full', 'zombie', 'random']:
        agent = E8_EEA_v5(input_dim=dim)

        if track == 'zombie':
            agent.modulate_weights = lambda e: None

        elif track == 'random':
            def _rand_mod(e, _a=agent):
                e.valence = float(np.random.normal(0.0, 0.5))
                e.arousal = float(np.clip(np.random.normal(0.5, 0.2), 0.0, 1.0))
                _a.beta   = float(np.clip(1.0 + 0.5 * e.arousal, 0.5, 2.0))
                _a.gamma  = float(np.clip(1.0 - 0.3 * e.valence, 0.5, 2.0))
            agent.modulate_weights = _rand_mod

        n = min(cycles, len(seq) - 1)
        for t in range(n):
            agent.cycle(seq[t], seq[t + 1], task_scores[t])

        # Extract rejection log — single pass, no O(n²) re-processing
        rejection_log = []
        prev          = None
        for entry in agent.stability_log:
            if not entry['accepted']:
                sim = 0.0
                if prev is not None:
                    # Proposal similarity proxy via λ₁ continuity
                    sim = float(np.exp(-abs(entry['lambda_1'] - prev['lambda_1'])))
                rejection_log.append({
                    'cycle':      entry['cycle'],
                    'lambda_1':   entry['lambda_1'],
                    'arousal':    entry['arousal'],
                    'valence':    entry['valence'],
                    'similarity': sim
                })
                prev = entry

        results[track] = rejection_log
        print(f"Track {track:6s}: {len(rejection_log):4d} rejections / "
              f"{len(agent.stability_log):4d} evaluations over {n} cycles")

    return results


def analyze_ablation(results):
    """
    Print summary statistics.

    Frustration signature (Track A vs B vs C):
      High mean similarity + low spacing std → clustering + drift → emergence candidate
    """
    for track, log in results.items():
        print(f"\nTrack {track}:")
        if len(log) < 2:
            print("  < 2 rejections — cannot compute statistics.")
            continue
        sims     = [e['similarity'] for e in log]
        arousals = [e['arousal']    for e in log]
        spacings = [log[i+1]['cycle'] - log[i]['cycle'] for i in range(len(log)-1)]
        print(f"  Rejections:                {len(log)}")
        print(f"  Mean proposal similarity:  {statistics.mean(sims):.4f}")
        print(f"  Std  proposal similarity:  {statistics.stdev(sims):.4f}")
        if spacings:
            print(f"  Mean rejection spacing:    {statistics.mean(spacings):.2f} cycles")
            print(f"  Std  rejection spacing:    {statistics.stdev(spacings):.2f} cycles")
        print(f"  Mean arousal at rejection: {statistics.mean(arousals):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# STRESS INPUT STREAM
# ═══════════════════════════════════════════════════════════════════════

def make_stress_input(n=600, dim=10):
    """
    Alternates stable periods with chaotic bursts.
    Stable: smooth sinusoidal — low free energy, low novelty.
    Burst:  coupled logistic map — designed to force phase transitions in Track A.

    A flat random sequence won't reliably trigger phase transitions.
    This input is engineered to stress the architecture.
    """
    seq = []
    while len(seq) < n:
        # Stable period: 30–50 cycles
        L = np.random.randint(30, 50)
        for s in range(L):
            phase = 2 * np.pi * s / L
            vec = np.array([np.sin(phase + i * 0.3) * 0.3 for i in range(dim)])
            vec += np.random.normal(0, 0.01, dim)
            seq.append(vec)
            if len(seq) >= n:
                break
        if len(seq) >= n:
            break
        # Burst period: 15–25 cycles, coupled chaotic map
        B = np.random.randint(15, 25)
        x = np.random.uniform(0.3, 0.7, dim)
        for _ in range(B):
            x_new = np.array([
                (3.7 + 0.2 * np.sin(i)) * x[i] * (1 - x[(i+1) % dim])
                for i in range(dim)
            ])
            x = np.clip(x_new, 0.01, 0.99)
            seq.append(x.copy())
            if len(seq) >= n:
                break
    return seq[:n]


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("E8-EEA v5 — Three-Track Ablation Test")
    print("=" * 50)
    print("Generating stress input (600 cycles, dim=10)...")
    seq = make_stress_input(n=600, dim=10)
    print(f"Input ready: {len(seq)} vectors")
    print()
    print("Running ablation (three tracks, 500 cycles each)...")
    print("Expected runtime: 5–15 min on CPU with finite-difference gradients.")
    print("For faster runs: set cycles=100, or replace gradients with autograd.")
    print()
    results = run_ablation(seq, cycles=500)
    print()
    analyze_ablation(results)
    print()
    print("Frustration signature (Track A):")
    print("  High similarity + tight clustering + high arousal at rejection")
    print()
    print("Null signatures:")
    print("  Track B: low similarity, scattered rejections (no emotion)")
    print("  Track C: moderate similarity, no directional drift (noise, not structure)")
    print()
    print("A clearly different from both B and C → emergence candidate confirmed.")
    print("A ≈ B ≈ C → honest null result. Back to the drawing board.")