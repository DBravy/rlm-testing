"""
Capital Retrieval with SVD Self-Configuration on DistilGPT-2

The model learns to dynamically configure its own weights via SVD direction
scaling to improve capital city retrieval. A regression head reads the model's
hidden state and outputs continuous scaling factors that are applied
differentiably so gradients flow through both the LoRA adapter and the
regression head simultaneously.

Architecture:
  1. Forward pass 1 (frozen): encode prompt, extract hidden state
  2. Regression head: hidden state -> scaling factors (differentiable)
  3. Forward pass 2 (with SVD hooks + LoRA): generate next token
  4. Loss: cross-entropy on correct capital token
  5. Gradients flow to both LoRA and regression head

Usage:
    python capital_retrieval.py
    python capital_retrieval.py --epochs 30 --lr_head 1e-3 --lr_lora 1e-4
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRAIN_PAIRS = [
    # Western Europe
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"),
    ("Austria", "Vienna"), ("Belgium", "Brussels"), ("Greece", "Athens"),
    ("Denmark", "Copenhagen"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Finland", "Helsinki"), ("Poland", "Warsaw"), ("Hungary", "Budapest"),
    ("Romania", "Bucharest"), ("Croatia", "Zagreb"), ("Serbia", "Belgrade"),
    ("Bulgaria", "Sofia"), ("Slovakia", "Bratislava"),
    # Eastern Europe / Central Asia
    ("Russia", "Moscow"), ("Ukraine", "Kiev"), ("Turkey", "Ankara"),
    ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"),
    # East Asia
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Taiwan", "Taipei"),
    ("Mongolia", "Ulaanbaatar"),
    # Southeast Asia
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
    ("Cambodia", "Phnom"), ("Myanmar", "Naypyidaw"),
    # South Asia
    ("India", "Delhi"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
    ("Nepal", "Kathmandu"), ("Sri Lanka", "Colombo"),
    # Middle East
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Israel", "Jerusalem"),
    ("Lebanon", "Beirut"), ("Jordan", "Amman"), ("Syria", "Damascus"),
    ("Yemen", "Sanaa"),
    # Africa
    ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Tanzania", "Dodoma"),
    ("Ghana", "Accra"), ("Senegal", "Dakar"), ("Uganda", "Kampala"),
    ("Zimbabwe", "Harare"), ("Mozambique", "Maputo"), ("Angola", "Luanda"),
    ("Tunisia", "Tunis"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"),
    # Americas
    ("Canada", "Ottawa"), ("Cuba", "Havana"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Ecuador", "Quito"), ("Bolivia", "La"),
    ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"),
    ("Panama", "Panama"), ("Jamaica", "Kingston"),
    ("Costa Rica", "San"), ("Guatemala", "Guatemala"),
    # Oceania
    ("Australia", "Canberra"), ("New Zealand", "Wellington"),
]

EVAL_PAIRS = [
    ("Brazil", "Brasilia"), ("Mexico", "Mexico"), ("Argentina", "Buenos"),
    ("South Korea", "Seoul"), ("Philippines", "Manila"),
    ("Colombia", "Bogota"), ("Venezuela", "Caracas"),
    ("Morocco", "Rabat"), ("Switzerland", "Bern"),
    ("Czech Republic", "Prague"), ("Malaysia", "Kuala"),
    ("Netherlands", "Amsterdam"), ("Saudi Arabia", "Riyadh"),
    ("Nigeria", "Abuja"), ("Ethiopia", "Addis"),
]


def build_prompt(country):
    return f"The capital of {country} is"


def get_target_token(tokenizer, capital):
    """Get the first token ID of the capital (with leading space)."""
    tokens = tokenizer.encode(f" {capital}", add_special_tokens=False)
    return tokens[0]


def verify_dataset(tokenizer, pairs, label=""):
    """Check which capitals are recognizable as first tokens."""
    valid = []
    for country, capital in pairs:
        tid = get_target_token(tokenizer, capital)
        decoded = tokenizer.decode([tid])
        valid.append((country, capital, tid, decoded.strip()))
    if label:
        print(f"\n{label}:")
        for country, capital, tid, decoded in valid:
            match = "ok" if decoded.lower().startswith(capital.lower()[:3]) else "MISMATCH"
            print(f"  {country}: '{capital}' -> token {tid} '{decoded}' [{match}]")
    return valid


# ---------------------------------------------------------------------------
# LoRA for GPT-2 Conv1D layers
# ---------------------------------------------------------------------------

class LoRAConv1D(nn.Module):
    """
    Wraps a GPT-2 Conv1D layer with a LoRA adapter.
    GPT-2 Conv1D computes: output = input @ weight + bias
    where weight has shape (in_features, out_features).
    """

    def __init__(self, original, r=8, alpha=16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features, out_features = original.weight.shape
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r

    def forward(self, x):
        # Original Conv1D: x @ W + b
        size_out = x.size()[:-1] + (self.original.nf,)
        out = torch.addmm(
            self.original.bias,
            x.view(-1, x.size(-1)),
            self.original.weight
        ).view(size_out)
        # LoRA delta
        lora_delta = (x @ self.lora_A @ self.lora_B) * self.scaling
        return out + lora_delta


def apply_lora(model, r=8, alpha=16):
    """Replace target Conv1D layers with LoRA-wrapped versions."""
    lora_params = []
    for i, block in enumerate(model.transformer.h):
        # Attention output projection
        block.attn.c_proj = LoRAConv1D(block.attn.c_proj, r=r, alpha=alpha)
        lora_params.extend([block.attn.c_proj.lora_A, block.attn.c_proj.lora_B])
        # MLP layers
        block.mlp.c_fc = LoRAConv1D(block.mlp.c_fc, r=r, alpha=alpha)
        lora_params.extend([block.mlp.c_fc.lora_A, block.mlp.c_fc.lora_B])
        block.mlp.c_proj = LoRAConv1D(block.mlp.c_proj, r=r, alpha=alpha)
        lora_params.extend([block.mlp.c_proj.lora_A, block.mlp.c_proj.lora_B])
    total = sum(p.numel() for p in lora_params)
    print(f"LoRA applied: {len(lora_params)} param tensors, {total:,} parameters")
    return lora_params


# ---------------------------------------------------------------------------
# SVD intervention system
# ---------------------------------------------------------------------------

class SVDInterventionManager:
    """
    Manages differentiable SVD interventions via forward hooks.

    Each intervention point is a (layer, matrix, direction) triple.
    During the modified forward pass, hooks add:
      delta = (alpha - 1) * sigma * (x @ u) * v
    to each target layer's output, where alpha comes from the regression head.
    """

    def __init__(self, model, num_directions=3):
        self.model = model
        self.intervention_points = []  # (layer_idx, matrix_name, dir_idx)
        self.svd_components = {}       # int_idx -> (u, sigma, v)
        self.hooks = []
        self.active = False
        # Mutable container so hooks can access current factors
        self._current_factors = [None]

        self._precompute(num_directions)
        self._register_hooks()

    def _get_conv1d_weight(self, layer_idx, matrix_name):
        """Get the ORIGINAL Conv1D weight (not LoRA)."""
        block = self.model.transformer.h[layer_idx]
        parts = matrix_name.split(".")
        obj = block
        for part in parts:
            obj = getattr(obj, part)
        # If wrapped with LoRA, get original weight
        if isinstance(obj, LoRAConv1D):
            return obj.original.weight
        return obj.weight

    def _precompute(self, num_directions):
        """Precompute SVDs and build intervention point list."""
        target_matrices = ["attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
        num_layers = len(self.model.transformer.h)

        print(f"Precomputing SVDs for {num_layers} layers...")
        idx = 0
        for layer_idx in range(num_layers):
            for matrix_name in target_matrices:
                try:
                    weight = self._get_conv1d_weight(layer_idx, matrix_name)
                    W = weight.detach().float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    for d in range(min(num_directions, len(S))):
                        self.intervention_points.append(
                            (layer_idx, matrix_name, d)
                        )
                        # For Conv1D: W is (in, out), U[:,d] is input dir,
                        # Vt[d,:] is output dir
                        self.svd_components[idx] = (
                            U[:, d].clone(),    # u: (in_features,)
                            S[d].clone(),       # sigma: scalar
                            Vt[d, :].clone(),   # v: (out_features,)
                        )
                        idx += 1
                except Exception as e:
                    print(f"  Warning: layer {layer_idx} {matrix_name}: {e}")

        print(f"  {len(self.intervention_points)} intervention points")

    def _get_module(self, layer_idx, matrix_name):
        """Get the module (possibly LoRA-wrapped) for hook registration."""
        block = self.model.transformer.h[layer_idx]
        parts = matrix_name.split(".")
        obj = block
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _register_hooks(self):
        """Register forward hooks grouped by (layer, matrix)."""
        # Group intervention points by module
        module_interventions = {}  # (layer, matrix) -> [(int_idx, u, sigma, v)]
        for int_idx, (layer_idx, matrix_name, dir_idx) in enumerate(
            self.intervention_points
        ):
            key = (layer_idx, matrix_name)
            if key not in module_interventions:
                module_interventions[key] = []
            u, sigma, v = self.svd_components[int_idx]
            module_interventions[key].append((int_idx, u, sigma, v))

        for (layer_idx, matrix_name), interventions in module_interventions.items():
            module = self._get_module(layer_idx, matrix_name)
            hook = self._make_hook(interventions)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)

    def _make_hook(self, interventions):
        """Create a forward hook for a specific module."""
        manager = self  # closure reference

        def hook(module, input_tuple, output):
            if not manager.active:
                return output
            factors = manager._current_factors[0]
            if factors is None:
                return output

            x = input_tuple[0]  # input to the Conv1D/LoRA layer
            delta = torch.zeros_like(output)

            for int_idx, u, sigma, v in interventions:
                alpha = factors[int_idx]
                u_dev = u.to(x.device, x.dtype)
                v_dev = v.to(x.device, x.dtype)
                s_dev = sigma.to(x.device, x.dtype)
                # Conv1D: y = x @ W, so delta_y = (alpha-1)*sigma*(x@u)*v
                proj = torch.einsum("...i,i->...", x, u_dev)
                delta = delta + (alpha - 1.0) * s_dev * proj.unsqueeze(-1) * v_dev

            return output + delta

        return hook

    def set_factors(self, factors):
        """Set scaling factors for the next forward pass."""
        self._current_factors[0] = factors

    @property
    def num_interventions(self):
        return len(self.intervention_points)


# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------

class ConfigHead(nn.Module):
    """
    Maps the model's hidden state to SVD scaling factors.
    Initialized to output 1.0 (no modification).
    """

    def __init__(self, hidden_dim, num_interventions):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_interventions)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  # centering ensures factors start at 1.0

    def forward(self, hidden_state):
        raw = self.linear(hidden_state)
        # Mean-center: forces selectivity. To amplify one direction,
        # the head must suppress another. Factors average to 1.0.
        centered = raw - raw.mean(dim=-1, keepdim=True)
        # tanh gives range [-1, 1], scale to [-2, 2], shift to [-1, 3]
        return 1.0 + 2.0 * torch.tanh(centered)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(model, config_head, svd_manager, tokenizer,
               country, capital, device):
    """
    One training step:
    1. Forward pass 1: get hidden state (no intervention)
    2. Config head: hidden -> scaling factors
    3. Forward pass 2: with SVD hooks active, get logits
    4. Loss: cross-entropy on target capital token
    """
    prompt = build_prompt(country)
    target_id = get_target_token(tokenizer, capital)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    target = torch.tensor([target_id], device=device)

    # Pass 1: get hidden state (hooks inactive)
    svd_manager.active = False
    with torch.no_grad():
        out1 = model(input_ids, output_hidden_states=True)
    hidden = out1.hidden_states[-1][0, -1, :]  # (hidden_dim,)

    # Config head: produce scaling factors (differentiable)
    scaling_factors = config_head(hidden)  # (num_interventions,)

    # Pass 2: with SVD hooks active
    svd_manager.set_factors(scaling_factors)
    svd_manager.active = True
    out2 = model(input_ids)
    logits = out2.logits[0, -1, :]  # (vocab_size,)
    svd_manager.active = False

    # Loss
    loss = F.cross_entropy(logits.unsqueeze(0), target)

    # Check prediction
    pred_id = logits.argmax().item()
    correct = (pred_id == target_id)

    return loss, correct, pred_id


def evaluate(model, config_head, svd_manager, tokenizer, pairs, device,
             use_svd=True):
    """Evaluate on a set of country-capital pairs."""
    model.eval()
    config_head.eval()
    correct = 0
    total = len(pairs)
    results = []

    with torch.no_grad():
        for country, capital in pairs:
            prompt = build_prompt(country)
            target_id = get_target_token(tokenizer, capital)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            if use_svd:
                # Pass 1: hidden state
                svd_manager.active = False
                out1 = model(input_ids, output_hidden_states=True)
                hidden = out1.hidden_states[-1][0, -1, :]
                factors = config_head(hidden)
                svd_manager.set_factors(factors)
                svd_manager.active = True

            out = model(input_ids)
            logits = out.logits[0, -1, :]
            svd_manager.active = False

            pred_id = logits.argmax().item()
            pred_token = tokenizer.decode([pred_id]).strip()
            is_correct = (pred_id == target_id)
            if is_correct:
                correct += 1
            results.append((country, capital, pred_token, is_correct))

    return correct, total, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading distilgpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad_(False)

    # Verify dataset
    print("\nVerifying dataset tokens...")
    verify_dataset(tokenizer, TRAIN_PAIRS[:5], "Train sample")
    verify_dataset(tokenizer, EVAL_PAIRS[:5], "Eval sample")

    # Apply LoRA (optional)
    lora_params = []
    if args.use_lora:
        print("\nApplying LoRA...")
        lora_params = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    else:
        print("\nSkipping LoRA (config head only)")

    # Build SVD intervention manager
    print("\nBuilding SVD intervention manager...")
    svd_manager = SVDInterventionManager(model, num_directions=args.num_directions)

    # Config head
    hidden_dim = model.config.n_embd
    config_head = ConfigHead(hidden_dim, svd_manager.num_interventions).to(device)
    print(f"Config head: {hidden_dim} -> {svd_manager.num_interventions} "
          f"({sum(p.numel() for p in config_head.parameters()):,} params)")

    # Optimizer
    if lora_params:
        optimizer = torch.optim.AdamW([
            {"params": config_head.parameters(), "lr": args.lr_head},
            {"params": lora_params, "lr": args.lr_lora},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            config_head.parameters(), lr=args.lr_head, weight_decay=0.01,
        )

    # --- Baseline evaluation ---
    print("\n" + "=" * 60)
    print("BASELINE (no LoRA, no SVD)")
    print("=" * 60)
    base_correct, base_total, base_results = evaluate(
        model, config_head, svd_manager, tokenizer, TRAIN_PAIRS, device,
        use_svd=False,
    )
    print(f"Train set: {base_correct}/{base_total} ({base_correct/base_total:.1%})")
    for country, capital, pred, ok in base_results:
        status = "ok" if ok else "MISS"
        print(f"  {country}: expected '{capital}', got '{pred}' [{status}]")

    eval_correct, eval_total, _ = evaluate(
        model, config_head, svd_manager, tokenizer, EVAL_PAIRS, device,
        use_svd=False,
    )
    print(f"Eval set:  {eval_correct}/{eval_total} ({eval_correct/eval_total:.1%})")

    # --- Training ---
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        config_head.train()

        epoch_loss = 0.0
        epoch_correct = 0
        pairs = TRAIN_PAIRS.copy()
        random.shuffle(pairs)

        for country, capital in pairs:
            optimizer.zero_grad()
            loss, correct, pred_id = train_step(
                model, config_head, svd_manager, tokenizer,
                country, capital, device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(config_head.parameters()) + lora_params, 1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            if correct:
                epoch_correct += 1

        avg_loss = epoch_loss / len(pairs)
        train_acc = epoch_correct / len(pairs)

        # Eval every few epochs
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            eval_c, eval_t, eval_results = evaluate(
                model, config_head, svd_manager, tokenizer,
                EVAL_PAIRS, device, use_svd=True,
            )
            eval_acc = eval_c / eval_t

            # Also check: what does eval look like WITHOUT SVD (LoRA only)?
            lora_c, lora_t, _ = evaluate(
                model, config_head, svd_manager, tokenizer,
                EVAL_PAIRS, device, use_svd=False,
            )
            lora_acc = lora_c / lora_t

            print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"train={train_acc:.1%} | "
                  f"eval(svd+lora)={eval_acc:.1%} | "
                  f"eval(lora only)={lora_acc:.1%}")
        else:
            print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"train={train_acc:.1%}")

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # Full system
    final_c, final_t, final_results = evaluate(
        model, config_head, svd_manager, tokenizer,
        EVAL_PAIRS, device, use_svd=True,
    )
    print(f"\nEval with SVD + LoRA: {final_c}/{final_t} ({final_c/final_t:.1%})")
    for country, capital, pred, ok in final_results:
        status = "ok" if ok else "MISS"
        print(f"  {country}: expected '{capital}', got '{pred}' [{status}]")

    # LoRA only (no SVD)
    lora_c, lora_t, _ = evaluate(
        model, config_head, svd_manager, tokenizer,
        EVAL_PAIRS, device, use_svd=False,
    )
    print(f"\nEval with LoRA only:  {lora_c}/{lora_t} ({lora_c/lora_t:.1%})")

    print(f"\nBaseline was:         {eval_correct}/{eval_total} "
          f"({eval_correct/eval_total:.1%})")

    # --- Show what the config head learned ---
    print("\n" + "=" * 60)
    print("CONFIG HEAD ANALYSIS")
    print("=" * 60)
    print("Average scaling factors per intervention point:")

    all_factors = []
    with torch.no_grad():
        for country, _ in TRAIN_PAIRS:
            prompt = build_prompt(country)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            svd_manager.active = False
            out = model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][0, -1, :]
            factors = config_head(hidden)
            all_factors.append(factors.cpu())

    stacked = torch.stack(all_factors)
    means = stacked.mean(dim=0)
    stds = stacked.std(dim=0)

    # Show the most deviated-from-1.0 intervention points
    deviations = (means - 1.0).abs()
    top_k = min(10, len(deviations))
    top_indices = deviations.topk(top_k).indices

    for idx in top_indices:
        layer, matrix, direction = svd_manager.intervention_points[idx]
        print(f"  layer {layer} {matrix} dir {direction}: "
              f"mean={means[idx]:.3f} std={stds[idx]:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_directions", type=int, default=3,
                        help="SVD directions per matrix")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
