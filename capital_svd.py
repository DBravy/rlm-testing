"""
Capital Retrieval with SVD Self-Configuration (distilgpt2)

Core components:
  - SVDManager: precomputes SVDs for GPT-2 weight matrices
  - VelocityAnalyzer: competitive logit lens showing per-component
    contributions to BOTH the target and the winner, plus gap analysis
  - Config generation: gap-based intervention targeting components
    that widen the gap between winner and target
  - Scaffold: rigid two-phase episode runner
"""

import re
import random
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Country-Capital dataset
# ---------------------------------------------------------------------------

CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"),
    ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Mexico", "Mexico"),
    ("Egypt", "Cairo"), ("India", "Delhi"), ("China", "Beijing"),
    ("Russia", "Moscow"), ("Turkey", "Ankara"), ("Thailand", "Bangkok"),
    ("Argentina", "Buenos"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
    ("Norway", "Oslo"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
    ("Portugal", "Lisbon"), ("Greece", "Athens"), ("Austria", "Vienna"),
    ("Switzerland", "Bern"), ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"), ("Ireland", "Dublin"), ("Hungary", "Budapest"),
    ("Romania", "Bucharest"), ("Colombia", "Bogota"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Cuba", "Havana"), ("Kenya", "Nairobi"),
    ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Morocco", "Rabat"),
    ("Iraq", "Baghdad"), ("Iran", "Tehran"), ("Israel", "Jerusalem"),
    ("Jordan", "Amman"), ("Lebanon", "Beirut"),
    ("Pakistan", "Islamabad"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
    ("Malaysia", "Kuala"), ("Singapore", "Singapore"),
    ("Croatia", "Zagreb"), ("Serbia", "Belgrade"),
    ("Ukraine", "Kyiv"), ("Czech Republic", "Prague"),
    ("Slovakia", "Bratislava"), ("Bulgaria", "Sofia"),
    ("South Korea", "Seoul"), ("North Korea", "Pyongyang"),
    ("New Zealand", "Wellington"), ("South Africa", "Pretoria"),
    ("Venezuela", "Caracas"), ("Ecuador", "Quito"),
    ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"),
    ("Bolivia", "La"), ("Panama", "Panama"),
    ("Costa Rica", "San"), ("Jamaica", "Kingston"),
    ("Iceland", "Reykjavik"), ("Luxembourg", "Luxembourg"),
    ("Malta", "Valletta"), ("Estonia", "Tallinn"),
    ("Latvia", "Riga"), ("Lithuania", "Vilnius"),
    ("Slovenia", "Ljubljana"), ("Albania", "Tirana"),
    ("Nepal", "Kathmandu"), ("Bangladesh", "Dhaka"),
    ("Myanmar", "Naypyidaw"), ("Cambodia", "Phnom"),
    ("Ethiopia", "Addis"), ("Tanzania", "Dodoma"),
    ("Uganda", "Kampala"), ("Senegal", "Dakar"),
    ("Tunisia", "Tunis"), ("Libya", "Tripoli"),
    ("Algeria", "Algiers"), ("Sudan", "Khartoum"),
]


def split_dataset(train_ratio=0.7, seed=42):
    rng = random.Random(seed)
    shuffled = list(CAPITALS)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


# ---------------------------------------------------------------------------
# SVD Manager (GPT-2 architecture)
# ---------------------------------------------------------------------------

class SVDManager:
    DEFAULT_MATRICES = [
        "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
    ]

    def __init__(self, model, matrix_names=None, max_directions=20):
        self.model = model
        self.matrix_names = matrix_names or self.DEFAULT_MATRICES
        self.max_directions = max_directions
        self.num_layers = len(model.transformer.h)
        self.svd_cache = {}
        self.active_deltas = []
        self._precompute_svds()

    def _get_weight(self, layer_idx, matrix_name):
        layer = self.model.transformer.h[layer_idx]
        parts = matrix_name.split(".")
        obj = layer
        for part in parts:
            obj = getattr(obj, part)
        return obj.weight

    def _precompute_svds(self):
        print(f"Precomputing SVDs for {self.num_layers} layers, "
              f"{len(self.matrix_names)} matrices each...")
        for layer_idx in range(self.num_layers):
            for matrix_name in self.matrix_names:
                try:
                    weight = self._get_weight(layer_idx, matrix_name)
                    W = weight.detach().float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    k = min(self.max_directions, len(S))
                    self.svd_cache[(layer_idx, matrix_name)] = (
                        U[:, :k].cpu(), S[:k].cpu(), Vt[:k, :].cpu(),
                    )
                except Exception as e:
                    print(f"  Warning: layer {layer_idx} {matrix_name}: {e}")
        print(f"  Cached {len(self.svd_cache)} decompositions")

    def scale_direction(self, layer_idx, matrix_name, direction_idx, scale_factor):
        key = (layer_idx, matrix_name)
        if key not in self.svd_cache:
            return None
        U, S, Vt = self.svd_cache[key]
        if direction_idx >= len(S):
            return None
        scale_factor = max(-5.0, min(scale_factor, 20.0))
        weight = self._get_weight(layer_idx, matrix_name)
        sigma = S[direction_idx].to(weight.dtype).to(weight.device)
        u = U[:, direction_idx].to(weight.dtype).to(weight.device)
        v = Vt[direction_idx, :].to(weight.dtype).to(weight.device)
        delta = (scale_factor - 1.0) * sigma * torch.outer(u, v)
        weight.data.add_(delta)
        self.active_deltas.append((weight, delta))
        return True

    def reset_all(self):
        count = len(self.active_deltas)
        for weight, delta in reversed(self.active_deltas):
            weight.data.sub_(delta.to(weight.device))
        self.active_deltas.clear()
        return count

    def get_spectrum(self, layer_idx, matrix_name):
        key = (layer_idx, matrix_name)
        if key not in self.svd_cache:
            return None
        _, S, _ = self.svd_cache[key]
        return S.tolist()

    def apply_config(self, config):
        for layer, matrix, direction, scale in config:
            self.scale_direction(layer, matrix, direction, scale)

    def random_config(self, num_scalings=3):
        config = []
        for _ in range(num_scalings):
            layer = random.randint(0, self.num_layers - 1)
            matrix = random.choice(self.matrix_names)
            direction = random.randint(0, min(self.max_directions, 10) - 1)
            scale = random.choice([0.0, 0.5, 1.5, 2.0, 3.0, 5.0])
            config.append((layer, matrix, direction, scale))
        return config


# ---------------------------------------------------------------------------
# Competitive Velocity Analyzer
# ---------------------------------------------------------------------------

class VelocityAnalyzer:
    """
    Velocity-based logit lens with competitive analysis.

    For each component (per-layer attention, MLP, per-head), computes:
      - Contribution to target token logit (the correct answer)
      - Contribution to winner token logit (what the model actually predicts)
      - Gap contribution = contrib_to_winner - contrib_to_target
        Positive gap = this component is widening the gap (hurting)
        Negative gap = this component is narrowing the gap (helping)

    Also provides triage: is the gap closeable?
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = len(model.transformer.h)
        self.unembed = model.lm_head.weight.detach().float()
        self.n_embd = model.config.n_embd
        self.n_head = model.config.n_head
        self.head_dim = self.n_embd // self.n_head

    def _get_token_id(self, token_str):
        ids_with_space = self.tokenizer.encode(" " + token_str)
        ids_without = self.tokenizer.encode(token_str)
        if len(ids_with_space) == 1:
            return ids_with_space[0]
        if len(ids_without) == 1:
            return ids_without[0]
        return ids_with_space[0]

    def analyze(self, prompt, target_token):
        """
        Full competitive analysis. Automatically identifies the winner
        and computes gap contributions for every component.

        Returns a dict with everything needed for intervention decisions.
        """
        target_id = self._get_token_id(target_token)
        target_dir = self.unembed[target_id].clone()

        # Hooks
        attn_velocities = {}
        mlp_velocities = {}
        c_proj_inputs = {}
        hooks = []

        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]

            def make_attn_hook(idx):
                def hook(module, inp, out):
                    attn_velocities[idx] = out[0].detach().float()
                return hook

            def make_mlp_hook(idx):
                def hook(module, inp, out):
                    mlp_velocities[idx] = out.detach().float()
                return hook

            def make_cproj_pre_hook(idx):
                def hook(module, inp):
                    c_proj_inputs[idx] = inp[0].detach().float()
                return hook

            hooks.append(block.attn.register_forward_hook(make_attn_hook(layer_idx)))
            hooks.append(block.mlp.register_forward_hook(make_mlp_hook(layer_idx)))
            hooks.append(block.attn.c_proj.register_forward_pre_hook(
                make_cproj_pre_hook(layer_idx)))

        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        for h in hooks:
            h.remove()

        # Final logits
        final_logits = outputs.logits[0, -1, :].float()

        # Identify winner
        winner_id = final_logits.argmax().item()
        winner_token = self.tokenizer.decode([winner_id]).strip()
        winner_dir = self.unembed[winner_id].clone()

        target_logit = final_logits[target_id].item()
        winner_logit = final_logits[winner_id].item()
        gap = winner_logit - target_logit  # positive = target is losing

        # Where does target rank?
        sorted_indices = final_logits.argsort(descending=True)
        target_rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item()

        # Top 5 predictions for context
        top5_ids = sorted_indices[:5].tolist()
        top5 = [
            (self.tokenizer.decode([tid]).strip(), final_logits[tid].item())
            for tid in top5_ids
        ]

        # Already correct?
        already_correct = (winner_id == target_id)

        # Triage computed after component analysis (see below)

        # Per-component analysis
        layer_results = []
        head_results = []

        for layer_idx in range(self.num_layers):
            attn_v = attn_velocities[layer_idx][0, -1, :]
            mlp_v = mlp_velocities[layer_idx][0, -1, :]

            attn_to_target = torch.dot(attn_v, target_dir).item()
            attn_to_winner = torch.dot(attn_v, winner_dir).item()
            mlp_to_target = torch.dot(mlp_v, target_dir).item()
            mlp_to_winner = torch.dot(mlp_v, winner_dir).item()

            # Gap contribution: how much this component widens the gap
            attn_gap = attn_to_winner - attn_to_target
            mlp_gap = mlp_to_winner - mlp_to_target

            layer_results.append({
                "layer": layer_idx,
                "attn_to_target": attn_to_target,
                "attn_to_winner": attn_to_winner,
                "attn_gap": attn_gap,
                "mlp_to_target": mlp_to_target,
                "mlp_to_winner": mlp_to_winner,
                "mlp_gap": mlp_gap,
            })

            # Per-head
            if layer_idx in c_proj_inputs:
                concat_heads = c_proj_inputs[layer_idx][0, -1, :]
                c_proj_w = self.model.transformer.h[
                    layer_idx
                ].attn.c_proj.weight.detach().float()

                for head_idx in range(self.n_head):
                    s = head_idx * self.head_dim
                    e = s + self.head_dim
                    head_input = concat_heads[s:e]
                    head_output = head_input @ c_proj_w[s:e, :]

                    h_to_target = torch.dot(head_output, target_dir).item()
                    h_to_winner = torch.dot(head_output, winner_dir).item()
                    h_gap = h_to_winner - h_to_target

                    head_results.append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "to_target": h_to_target,
                        "to_winner": h_to_winner,
                        "gap": h_gap,
                    })

        # --- Triage: is this closeable? ---
        # Based on whether gap offenders have enough leverage.
        # EXCLUDE the final layer: it amplifies decisions already made
        # by earlier layers, so suppressing it doesn't fix the root cause.
        all_layer_indices = [lr["layer"] for lr in layer_results]
        final_layer = max(all_layer_indices) if all_layer_indices else 0

        if already_correct:
            closeable = True
            leverage_ratio = 0.0
        elif gap <= 0:
            closeable = True
            leverage_ratio = float("inf")
        else:
            # Collect gap contributions from actionable layers only
            actionable_gaps = []
            for lr in layer_results:
                if lr["layer"] < final_layer:
                    actionable_gaps.append(lr["attn_gap"])
                    actionable_gaps.append(lr["mlp_gap"])
            for hr in head_results:
                if hr["layer"] < final_layer:
                    actionable_gaps.append(hr["gap"])

            actionable_gaps.sort(reverse=True)

            worst_single = actionable_gaps[0] if actionable_gaps else 0.0
            top3_sum = (sum(actionable_gaps[:3])
                        if len(actionable_gaps) >= 3 else sum(actionable_gaps))

            leverage_ratio = worst_single / gap if gap > 0 else 0.0

            closeable = (target_rank < 30 and
                        ((leverage_ratio >= 1.0) or (top3_sum >= 2.0 * gap)))

        return {
            "target_token": target_token,
            "target_id": target_id,
            "target_logit": target_logit,
            "target_rank": target_rank,
            "winner_token": winner_token,
            "winner_id": winner_id,
            "winner_logit": winner_logit,
            "gap": gap,
            "already_correct": already_correct,
            "closeable": closeable,
            "leverage_ratio": leverage_ratio,
            "num_layers": self.num_layers,
            "top5": top5,
            "layer_results": layer_results,
            "head_results": head_results,
        }


# ---------------------------------------------------------------------------
# Gap-based config generation
# ---------------------------------------------------------------------------

def generate_targeted_configs(analysis, svd_manager, num_configs=30,
                              directions_to_try=(0, 1, 2),
                              suppress_scales=(0.0, 0.3, 0.5),
                              amplify_scales=(1.5, 2.0, 3.0)):
    """
    Generate SVD configurations based on competitive gap analysis.

    Primary strategy: suppress components that WIDEN the gap (large positive
    gap contribution = pushing toward winner more than toward target).

    Secondary strategy: amplify components that NARROW the gap (negative
    gap contribution = pushing toward target more than winner).

    Combo strategy: suppress worst gap offender + amplify best gap helper.
    """
    if analysis["already_correct"]:
        return []  # nothing to fix

    if not analysis["closeable"]:
        return []  # triage: not worth trying

    # Collect all components with their gap contributions
    # EXCLUDE the final layer: it amplifies prior decisions, not the root cause
    all_layers = [lr["layer"] for lr in analysis["layer_results"]]
    final_layer = max(all_layers) if all_layers else 0
    components = []
    for lr in analysis["layer_results"]:
        layer = lr["layer"]
        if layer >= final_layer:
            continue
        components.append({
            "layer": layer, "type": "attn", "matrix": "attn.c_proj",
            "gap": lr["attn_gap"],
            "to_target": lr["attn_to_target"],
            "to_winner": lr["attn_to_winner"],
        })
        components.append({
            "layer": layer, "type": "mlp_down", "matrix": "mlp.c_proj",
            "gap": lr["mlp_gap"],
            "to_target": lr["mlp_to_target"],
            "to_winner": lr["mlp_to_winner"],
        })
        components.append({
            "layer": layer, "type": "mlp_up", "matrix": "mlp.c_fc",
            "gap": lr["mlp_gap"],
            "to_target": lr["mlp_to_target"],
            "to_winner": lr["mlp_to_winner"],
        })

    # Also include per-head components (excluding final layer)
    for hr in analysis["head_results"]:
        if hr["layer"] >= final_layer:
            continue
        components.append({
            "layer": hr["layer"], "type": f"head_{hr['head']}",
            "matrix": "attn.c_proj",
            "gap": hr["gap"],
            "to_target": hr["to_target"],
            "to_winner": hr["to_winner"],
        })

    # Sort by gap contribution (largest positive = worst offender)
    components.sort(key=lambda c: c["gap"], reverse=True)

    configs = []

    # Strategy 1: Suppress worst gap offenders (components widening the gap)
    worst_offenders = [c for c in components if c["gap"] > 0][:6]
    for comp in worst_offenders:
        for d in directions_to_try:
            for s in suppress_scales:
                configs.append([
                    (comp["layer"], comp["matrix"], d, s)
                ])

    # Strategy 2: Amplify best gap helpers (components narrowing the gap)
    best_helpers = [c for c in components if c["gap"] < 0]
    best_helpers.sort(key=lambda c: c["gap"])  # most negative first
    for comp in best_helpers[:4]:
        for d in directions_to_try:
            for s in amplify_scales:
                configs.append([
                    (comp["layer"], comp["matrix"], d, s)
                ])

    # Strategy 3: Combo: suppress worst + amplify best
    if worst_offenders and best_helpers:
        for wo in worst_offenders[:3]:
            for bh in best_helpers[:3]:
                # Skip if same layer and matrix (redundant)
                if wo["layer"] == bh["layer"] and wo["matrix"] == bh["matrix"]:
                    continue
                for wd in directions_to_try[:2]:
                    for bd in directions_to_try[:2]:
                        configs.append([
                            (wo["layer"], wo["matrix"], wd, 0.0),
                            (bh["layer"], bh["matrix"], bd, 2.0),
                        ])

    # Strategy 4: Suppress components that strongly promote the winner
    # (even if they also somewhat promote the target)
    winner_promoters = sorted(
        components, key=lambda c: c["to_winner"], reverse=True
    )[:4]
    for comp in winner_promoters:
        if comp["to_winner"] > 0:
            for d in directions_to_try[:2]:
                for s in suppress_scales:
                    configs.append([
                        (comp["layer"], comp["matrix"], d, s)
                    ])

    # Deduplicate
    seen = set()
    unique = []
    for cfg in configs:
        key = tuple(tuple(x) for x in cfg)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)

    random.shuffle(unique)
    return unique[:num_configs]


# ---------------------------------------------------------------------------
# Config text format
# ---------------------------------------------------------------------------

def config_to_text(config):
    return "; ".join(
        f"{layer},{matrix},{direction},{scale}"
        for layer, matrix, direction, scale in config
    )


def text_to_config(text):
    config = []
    for part in re.split(r"[;\n]", text):
        part = part.strip()
        if not part:
            continue
        match = re.match(
            r"(\d+)\s*,\s*([\w.]+)\s*,\s*(\d+)\s*,\s*(-?[\d.]+)", part
        )
        if match:
            config.append((
                int(match.group(1)), match.group(2),
                int(match.group(3)), float(match.group(4)),
            ))
    return config


# ---------------------------------------------------------------------------
# Episode data
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    country: str
    gold_capital: str
    config: list = field(default_factory=list)
    config_text: str = ""
    prediction: str = None
    correct: bool = False
    baseline_prediction: str = None
    baseline_correct: bool = False
    velocity_guided: bool = False
    gap: float = 0.0
    target_rank: int = -1
    closeable: bool = False


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------

def build_config_prompt(country):
    return (
        "Country: France\n"
        "Config: 3,mlp.c_fc,2,1.5; 5,attn.c_proj,0,0.8\n"
        "Country: Japan\n"
        "Config: 4,attn.c_attn,1,2.0; 2,mlp.c_proj,3,0.5\n"
        "Country: Germany\n"
        "Config: 1,mlp.c_fc,0,1.3; 4,attn.c_proj,2,1.5; 5,mlp.c_proj,1,0.7\n"
        f"Country: {country}\n"
        "Config:"
    )


def build_capital_prompt(country):
    return f"The capital of {country} is"


class Scaffold:
    def __init__(self, model, tokenizer, svd_manager,
                 config_max_tokens=60):
        self.model = model
        self.tokenizer = tokenizer
        self.svd_manager = svd_manager
        self.config_max_tokens = config_max_tokens

    def _get_top_token(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        top_id = logits.argmax().item()
        return self.tokenizer.decode([top_id]).strip()

    def _generate(self, prompt, max_new_tokens, temperature=0.8):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _check_capital(self, pred, gold):
        if not pred or not gold:
            return False
        return (pred.lower().startswith(gold.lower()) or
                gold.lower().startswith(pred.lower()))

    def baseline_eval(self, country, gold_capital):
        self.svd_manager.reset_all()
        pred = self._get_top_token(build_capital_prompt(country))
        correct = self._check_capital(pred, gold_capital)
        return pred, correct

    def run_episode(self, country, gold_capital, forced_config=None,
                    run_baseline=True):
        episode = Episode(country=country, gold_capital=gold_capital)

        if run_baseline:
            base_pred, base_ok = self.baseline_eval(country, gold_capital)
            episode.baseline_prediction = base_pred
            episode.baseline_correct = base_ok

        if forced_config is not None:
            config = forced_config
        else:
            raw = self._generate(
                build_config_prompt(country), self.config_max_tokens, 0.8
            )
            config = text_to_config(raw.split("\n")[0].strip())

        episode.config = config
        episode.config_text = config_to_text(config)

        self.svd_manager.reset_all()
        self.svd_manager.apply_config(config)

        pred = self._get_top_token(build_capital_prompt(country))
        episode.prediction = pred
        episode.correct = self._check_capital(pred, gold_capital)

        self.svd_manager.reset_all()
        return episode


# ---------------------------------------------------------------------------
# SFT training data
# ---------------------------------------------------------------------------

def episode_to_training_text(episode):
    config_text = config_to_text(episode.config) if episode.config else ""
    return build_config_prompt(episode.country) + f" {config_text}\n"
