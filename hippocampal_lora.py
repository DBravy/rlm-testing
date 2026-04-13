"""
Hippocampal LoRA: Residual Stream -> Hippocampus -> Activation Injection

The hippocampus:
  - Receives residual stream states as "cortical input" via EC superficial
  - Sparsifies through DG into CA3
  - Stores temporal associations via Hebbian learning in CA3
  - On recall: pattern-completes in CA3, reconstructs via CA1/subiculum
  - Injects the reconstructed state back into the residual stream

The hippocampus learns LOCALLY (Hebbian only). No gradients flow through it.
It observes what cortex does during training and replays it during inference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Hippocampal Components (adapted from user's model)
# ---------------------------------------------------------------------------

def make_feedforward_weights(D_output, d_input, connectivity_prob=0.33,
                             device='cpu', dtype=torch.float32):
    mask = (torch.rand(D_output, d_input, device=device) < connectivity_prob).to(dtype)
    weights = torch.randn(D_output, d_input, device=device, dtype=dtype) * mask
    row_norms = torch.linalg.norm(weights, dim=1, keepdim=True) + 1e-10
    return weights / row_norms


def build_ring_inhibition(D, sigma, connection_prob=None,
                          device='cpu', dtype=torch.float32):
    positions = torch.arange(D, device=device, dtype=dtype)
    dist = torch.abs(positions[:, None] - positions[None, :])
    dist = torch.minimum(dist, D - dist)
    W = torch.exp(-dist**2 / (2 * sigma**2))
    W[dist > 3 * sigma] = 0
    W.fill_diagonal_(0)
    if connection_prob is not None:
        mask = (torch.rand(D, D, device=device) < connection_prob).to(dtype)
        mask.fill_diagonal_(0)
        W *= mask
    row_sums = W.sum(dim=1, keepdim=True) + 1e-10
    W /= row_sums
    return W


def apply_kwta(pattern, k):
    out = torch.zeros_like(pattern)
    if k < pattern.shape[0]:
        values, indices = torch.topk(pattern, k)
        out[indices] = values
    else:
        out = pattern.clone()
    return out


def cosine_sim(a, b):
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


class ECSuperficial:
    """Receives cortical input, produces stellate (sparse) and pyramidal (dense)."""

    def __init__(self, d_ec, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
                 pyr_to_stel_strength=0.3, connectivity_prob=0.33,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.pyr_to_stel_strength = pyr_to_stel_strength
        self.W_stellate = make_feedforward_weights(d_ec, d_ec, connectivity_prob, device, dtype)
        self.W_inh = build_ring_inhibition(d_ec, sigma_inh, device=device, dtype=dtype)
        self.W_pyramidal = make_feedforward_weights(d_ec, d_ec, connectivity_prob, device, dtype)
        self.W_pyr_to_stel = make_feedforward_weights(d_ec, d_ec, connectivity_prob, device, dtype)

    def forward(self, cortical_input):
        pyramidal = torch.relu(self.W_pyramidal @ cortical_input)
        h_raw = self.W_stellate @ cortical_input
        h_raw = h_raw + self.pyr_to_stel_strength * (self.W_pyr_to_stel @ pyramidal)
        h_raw = torch.relu(h_raw)
        h = h_raw.clone()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = torch.relu(h_raw - self.gamma_inh * inh)
        return h, pyramidal


class DentateGyrusLateral:
    """Sparsifies EC input for pattern separation."""

    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
                 n_inh_steps=5, noise_scale=0.0,
                 device='cpu', dtype=torch.float32):
        self.D_output = D_output
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.noise_scale = noise_scale
        self.device = device
        self.dtype = dtype
        self.W_ff = make_feedforward_weights(D_output, d_input, device=device, dtype=dtype)
        self.W_inh = build_ring_inhibition(D_output, sigma_inh, device=device, dtype=dtype)

    def forward(self, x):
        h_raw = x @ self.W_ff.T
        h_raw = torch.relu(h_raw)
        if self.noise_scale > 0 and torch.any(h_raw > 0):
            mean_active = h_raw[h_raw > 0].mean()
            h_raw = torch.relu(
                h_raw + torch.randn(self.D_output, device=self.device,
                                    dtype=self.dtype) * self.noise_scale * mean_active)
        h = h_raw.clone()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = torch.relu(h_raw - self.gamma_inh * inh)
        return h


class CA3Temporal:
    """Auto-associative network with temporal (successor) associations."""

    def __init__(self, N, k_active, lr=1.0, device='cpu', dtype=torch.float32):
        self.N = N
        self.k_active = k_active
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.W = torch.zeros((N, N), device=device, dtype=dtype)
        self.n_stored = 0
        self.mean_activity = torch.zeros(N, device=device, dtype=dtype)

    def _normalize_and_center(self, pattern):
        p = pattern / (torch.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        return p, p_c

    def store_online(self, pattern, prev_pattern=None):
        _, curr_c = self._normalize_and_center(pattern)
        if prev_pattern is not None:
            prev_p = prev_pattern / (torch.linalg.norm(prev_pattern) + 1e-10)
            prev_c = prev_p - self.mean_activity
            self.W += self.lr * torch.outer(curr_c, prev_c)
            self.W.fill_diagonal_(0)

    def retrieve(self, cue, n_iterations=5):
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        for _ in range(n_iterations):
            h = torch.relu(self.W @ x)
            x_new = apply_kwta(h, self.k_active)
            norm = torch.linalg.norm(x_new)
            if norm < 1e-10:
                break
            x_new = x_new / norm
            if torch.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x


class CA1:
    """Learns to reconstruct EC stellate from CA3, with EC input during encoding."""

    def __init__(self, N_ca1, N_ca3, lr=0.3, weight_decay=0.998, k_active=50,
                 device='cpu', dtype=torch.float32):
        self.N_ca1 = N_ca1
        self.lr = lr
        self.weight_decay = weight_decay
        self.k_active = k_active
        self.W_sc = torch.zeros((N_ca1, N_ca3), device=device, dtype=dtype)

    def encode(self, x_ca3, x_ec_stel):
        self.W_sc += self.lr * torch.outer(x_ec_stel, x_ca3)
        self.W_sc *= self.weight_decay

    def retrieve(self, x_ca3):
        h_sc = torch.relu(self.W_sc @ x_ca3)
        return apply_kwta(h_sc, self.k_active)


class Subiculum:
    """Reconstructs dense (pyramidal) component from CA1 output."""

    def __init__(self, N_sub, N_ca1, lr=1.0, k_active=500,
                 device='cpu', dtype=torch.float32):
        self.N_sub = N_sub
        self.lr = lr
        self.k_active = k_active
        self.W_ca1 = torch.zeros((N_sub, N_ca1), device=device, dtype=dtype)

    def encode(self, ca1_output, ec_pyramidal):
        self.W_ca1 += self.lr * torch.outer(ec_pyramidal, ca1_output)

    def replay(self, ca1_output):
        h = torch.relu(self.W_ca1 @ ca1_output)
        return apply_kwta(h, self.k_active)


# ---------------------------------------------------------------------------
# Cortex-Hippocampus Interface
# ---------------------------------------------------------------------------

class CortexToHippoProjection:
    """
    Fixed random projection from cortical space (residual stream dim)
    to EC space. Analogous to the cortex -> EC pathway.
    Not learned. Just dimensionality adaptation.
    """

    def __init__(self, d_cortex, d_ec, device='cpu', dtype=torch.float32):
        self.d_cortex = d_cortex
        self.d_ec = d_ec
        if d_cortex == d_ec:
            self.W_down = None  # identity
            self.W_up = None
        else:
            # Random projection down (cortex -> EC)
            W = torch.randn(d_ec, d_cortex, device=device, dtype=dtype)
            W = W / (torch.linalg.norm(W, dim=1, keepdim=True) + 1e-10)
            self.W_down = W
            # Pseudoinverse for reconstruction (EC -> cortex)
            self.W_up = torch.linalg.pinv(W)

    def project_down(self, cortical_state):
        """Cortex -> EC space."""
        if self.W_down is None:
            return cortical_state
        return self.W_down @ cortical_state

    def project_up(self, ec_state):
        """EC space -> cortex."""
        if self.W_up is None:
            return ec_state
        return self.W_up @ ec_state


# ---------------------------------------------------------------------------
# Full Hippocampal System with Cortical Interface
# ---------------------------------------------------------------------------

class HippocampalSystem:
    """
    Full hippocampal system that interfaces with a transformer's residual stream.

    Encoding: receives residual stream states during training, stores trajectories.
    Retrieval: given a residual stream cue, recalls the nearest trajectory and
               produces an activation to inject back into the stream.
    """

    def __init__(self, d_cortex, d_ec=None, D_dg=None, N_ca3=None, k_ca3=50,
                 N_ca1=None, N_sub=None,
                 ca3_lr=1.0, ca1_lr=50.0, ca1_decay=1.0, sub_lr=1.0,
                 ca3_retrieval_iters=5,
                 device='cpu', dtype=torch.float32):
        if d_ec is None:
            d_ec = min(d_cortex, 384)
        if D_dg is None:
            D_dg = d_ec
        if N_ca3 is None:
            N_ca3 = d_ec
        if N_ca1 is None:
            N_ca1 = d_ec
        if N_sub is None:
            N_sub = d_ec

        self.d_cortex = d_cortex
        self.d_ec = d_ec
        self.ca3_retrieval_iters = ca3_retrieval_iters
        self.device = device
        self.dtype = dtype

        # Cortex <-> EC projection
        self.projection = CortexToHippoProjection(d_cortex, d_ec, device, dtype)

        # Hippocampal regions
        self.ec_sup = ECSuperficial(
            d_ec, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
            device=device, dtype=dtype)
        self.dg = DentateGyrusLateral(
            d_ec, D_dg, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
            device=device, dtype=dtype)
        self.ca3 = CA3Temporal(N_ca3, k_ca3, lr=ca3_lr, device=device, dtype=dtype)
        self.ca1 = CA1(N_ca1, N_ca3, lr=ca1_lr, weight_decay=ca1_decay,
                       device=device, dtype=dtype)
        self.sub = Subiculum(N_sub, N_ca1, lr=sub_lr, device=device, dtype=dtype)

        # Direct pathway (EC -> CA3, bypassing DG)
        self.W_direct = torch.zeros((N_ca3, d_ec), device=device, dtype=dtype)
        self.direct_lr = 0.3
        self.direct_decay = 0.998
        self.k_ca3 = k_ca3

        # Sequence tracking
        self._prev_dg_pattern = None
        self._in_sequence = False

        # Stats
        self.n_encoded = 0
        self.n_sequences = 0

    def begin_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = True
        self.n_sequences += 1

    def end_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = False

    def encode(self, cortical_state):
        """
        Encode a single cortical state (residual stream snapshot).
        Call this sequentially during training.
        """
        # Move to hippocampal device (model may be on GPU, hippocampus on CPU)
        cortical_state = cortical_state.detach().to(self.device, self.dtype)

        # Project down to EC space
        ec_input = self.projection.project_down(cortical_state)

        # EC superficial processing
        stellate, pyramidal = self.ec_sup.forward(ec_input)

        # DG sparsification
        dg_out = self.dg.forward(stellate)

        # CA3 temporal storage (Hebbian)
        prev = self._prev_dg_pattern if self._in_sequence else None
        self.ca3.store_online(dg_out, prev_pattern=prev)
        if self._in_sequence:
            self._prev_dg_pattern = dg_out.clone()

        # Direct pathway learning
        dg_norm = dg_out / (torch.linalg.norm(dg_out) + 1e-10)
        stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
        self.W_direct += self.direct_lr * torch.outer(dg_norm, stel_norm)
        self.W_direct *= self.direct_decay

        # CA1 learns reconstruction
        self.ca1.encode(dg_out, stellate)

        # Subiculum learns dense reconstruction
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iters)
        ca1_out = self.ca1.retrieve(ca3_out)
        self.sub.encode(ca1_out, pyramidal)

        self.n_encoded += 1
        return dg_out

    def recall(self, cortical_state):
        """
        Given a cortical state (residual stream snapshot), recall
        from hippocampus and produce an activation to inject.

        Returns the recalled activation in cortical space (d_cortex).
        """
        # Move to hippocampal device
        cortical_state = cortical_state.detach().to(self.device, self.dtype)

        # Project down to EC space
        ec_input = self.projection.project_down(cortical_state)

        # EC superficial
        stellate, pyramidal = self.ec_sup.forward(ec_input)

        # Cue CA3 via direct pathway + DG
        raw_dg = self.dg.forward(stellate)
        raw_direct = torch.relu(self.W_direct @ stellate)
        combined = raw_dg + raw_direct
        ca3_cue = apply_kwta(combined, self.k_ca3)
        norm = torch.linalg.norm(ca3_cue)
        if norm > 1e-10:
            ca3_cue = ca3_cue / norm

        # CA3 pattern completion
        ca3_out = self.ca3.retrieve(ca3_cue, self.ca3_retrieval_iters)

        # CA1 reconstruction (sparse component)
        ca1_out = self.ca1.retrieve(ca3_out)

        # Subiculum reconstruction (dense component)
        sub_out = self.sub.replay(ca1_out)

        # Combine via EC deep gating
        gamma = torch.sigmoid(sub_out)
        ec_deep_out = gamma * ca1_out

        # Project back up to cortical space
        cortical_output = self.projection.project_up(ec_deep_out)

        return cortical_output

    def recall_similarity(self, cortical_state):
        """
        Recall and return both the injection and a similarity score
        indicating how confident the recall is.
        """
        recalled = self.recall(cortical_state)
        # Similarity between cue and recall (rough confidence measure)
        cortical_state = cortical_state.detach().to(self.device, self.dtype)
        cue_norm = cortical_state / (torch.linalg.norm(cortical_state) + 1e-10)
        rec_norm = recalled / (torch.linalg.norm(recalled) + 1e-10)
        sim = float(torch.dot(cue_norm, rec_norm))
        return recalled, sim


# ---------------------------------------------------------------------------
# Transformer Hook System
# ---------------------------------------------------------------------------

class ResidualStreamHook:
    """
    Hooks into a transformer layer to capture and optionally inject
    into the residual stream.
    """

    def __init__(self, target_layer_idx):
        self.target_layer_idx = target_layer_idx
        self.captured_state = None
        self.injection = None  # set this to inject
        self.injection_strength = 1.0
        self._hook = None

    def _hook_fn(self, module, inp, out):
        # Transformer block output is the residual stream after this layer
        # For GPT-2: output is a tuple, first element is hidden states
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out

        # Capture the state at the last token position
        self.captured_state = hidden[0, -1, :].detach().clone()

        # Inject if we have something to inject
        if self.injection is not None:
            injection = self.injection.to(hidden.device).to(hidden.dtype)
            # Add injection to the last token position
            if isinstance(out, tuple):
                modified = list(out)
                h = modified[0].clone()
                h[0, -1, :] += self.injection_strength * injection
                modified[0] = h
                return tuple(modified)
            else:
                hidden = hidden.clone()
                hidden[0, -1, :] += self.injection_strength * injection
                return hidden

    def attach(self, model, layer_idx=None):
        """Attach hook to a transformer layer."""
        idx = layer_idx if layer_idx is not None else self.target_layer_idx
        layer = model.transformer.h[idx]
        self._hook = layer.register_forward_hook(self._hook_fn)

    def detach(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def clear(self):
        self.captured_state = None
        self.injection = None


# ---------------------------------------------------------------------------
# Knowledge Domains
# ---------------------------------------------------------------------------

DOMAINS = {
    "capitals": [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of Canada is", "Ottawa"),
        ("The capital of Egypt is", "Cairo"),
        ("The capital of China is", "Beijing"),
        ("The capital of Russia is", "Moscow"),
        ("The capital of Thailand is", "Bangkok"),
        ("The capital of Poland is", "Warsaw"),
        ("The capital of Sweden is", "Stockholm"),
        ("The capital of Norway is", "Oslo"),
        ("The capital of Greece is", "Athens"),
        ("The capital of Austria is", "Vienna"),
        ("The capital of Portugal is", "Lisbon"),
        ("The capital of Ireland is", "Dublin"),
        ("The capital of Hungary is", "Budapest"),
        ("The capital of Peru is", "Lima"),
        ("The capital of Chile is", "Santiago"),
    ],
    "elements": [
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for silver is", "Ag"),
        ("The chemical symbol for iron is", "Fe"),
        ("The chemical symbol for copper is", "Cu"),
        ("The chemical symbol for sodium is", "Na"),
        ("The chemical symbol for potassium is", "K"),
        ("The chemical symbol for calcium is", "Ca"),
        ("The chemical symbol for oxygen is", "O"),
        ("The chemical symbol for hydrogen is", "H"),
        ("The chemical symbol for nitrogen is", "N"),
        ("The chemical symbol for carbon is", "C"),
        ("The chemical symbol for helium is", "He"),
        ("The chemical symbol for lead is", "Pb"),
        ("The chemical symbol for tin is", "Sn"),
        ("The chemical symbol for mercury is", "Hg"),
        ("The chemical symbol for zinc is", "Zn"),
        ("The chemical symbol for neon is", "Ne"),
        ("The chemical symbol for argon is", "Ar"),
        ("The chemical symbol for chlorine is", "Cl"),
        ("The chemical symbol for fluorine is", "F"),
    ],
    "languages": [
        ("The official language of France is", "French"),
        ("The official language of Japan is", "Japanese"),
        ("The official language of Germany is", "German"),
        ("The official language of Brazil is", "Portuguese"),
        ("The official language of China is", "Mandarin"),
        ("The official language of Russia is", "Russian"),
        ("The official language of Italy is", "Italian"),
        ("The official language of Egypt is", "Arabic"),
        ("The official language of Thailand is", "Thai"),
        ("The official language of Greece is", "Greek"),
        ("The official language of Poland is", "Polish"),
        ("The official language of Sweden is", "Swedish"),
        ("The official language of Turkey is", "Turkish"),
        ("The official language of Iran is", "Persian"),
        ("The official language of Israel is", "Hebrew"),
        ("The official language of South Korea is", "Korean"),
        ("The official language of Vietnam is", "Vietnamese"),
        ("The official language of Netherlands is", "Dutch"),
        ("The official language of Finland is", "Finnish"),
        ("The official language of Denmark is", "Danish"),
    ],
    "opposites": [
        ("The opposite of hot is", "cold"),
        ("The opposite of big is", "small"),
        ("The opposite of fast is", "slow"),
        ("The opposite of light is", "dark"),
        ("The opposite of happy is", "sad"),
        ("The opposite of old is", "young"),
        ("The opposite of rich is", "poor"),
        ("The opposite of strong is", "weak"),
        ("The opposite of tall is", "short"),
        ("The opposite of loud is", "quiet"),
        ("The opposite of hard is", "soft"),
        ("The opposite of wet is", "dry"),
        ("The opposite of early is", "late"),
        ("The opposite of clean is", "dirty"),
        ("The opposite of sharp is", "dull"),
        ("The opposite of thick is", "thin"),
        ("The opposite of deep is", "shallow"),
        ("The opposite of sweet is", "bitter"),
        ("The opposite of smooth is", "rough"),
        ("The opposite of wide is", "narrow"),
    ],
    "colors": [
        ("The color of grass is", "green"),
        ("The color of the sky is", "blue"),
        ("The color of blood is", "red"),
        ("The color of snow is", "white"),
        ("The color of coal is", "black"),
        ("The color of the sun is", "yellow"),
        ("The color of an orange is", "orange"),
        ("The color of a violet is", "purple"),
        ("The color of chocolate is", "brown"),
        ("The color of a flamingo is", "pink"),
        ("The color of silver is", "gray"),
        ("The color of gold is", "golden"),
        ("The color of the ocean is", "blue"),
        ("The color of a ruby is", "red"),
        ("The color of an emerald is", "green"),
        ("The color of ivory is", "white"),
        ("The color of charcoal is", "black"),
        ("The color of rust is", "orange"),
        ("The color of lavender is", "purple"),
        ("The color of sand is", "tan"),
    ],
}


def domain_to_training_texts(domain_name):
    return [f"{prompt} {answer}" for prompt, answer in DOMAINS[domain_name]]
