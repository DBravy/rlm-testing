"""
SVD Direction Scaling Exploration for Qwen3-0.6B

Brute-force search over SVD direction scalings across layers and weight matrices.
For each configuration, evaluates math performance on a sample of GSM8K problems
and reports which directions/layers/scaling factors actually move the needle.

Usage:
    python svd_exploration.py
    python svd_exploration.py --num_problems 50 --num_directions 10
    python svd_exploration.py --layers 4 5 6 --matrices attn.o_proj mlp.up_proj
"""

import argparse
import re
import json
import time
from pathlib import Path
from itertools import product

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_NUM_PROBLEMS = 30        # GSM8K problems to evaluate per configuration
DEFAULT_NUM_DIRECTIONS = 5       # top-k singular directions to try per matrix
DEFAULT_MAX_NEW_TOKENS = 512
SCALING_FACTORS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]

# Weight matrices to explore. These are the names as they appear inside each
# transformer layer of Qwen3 (model.layers.{i}.{matrix_name}.weight)
DEFAULT_MATRICES = [
    "self_attn.o_proj",      # attention output projection
    "self_attn.q_proj",      # query projection
    "mlp.up_proj",           # MLP up-projection (SwiGLU)
    "mlp.down_proj",         # MLP down-projection
]

# Which layers to sweep. None means auto-select a spread across the model.
DEFAULT_LAYERS = None


# ---------------------------------------------------------------------------
# SVD utilities
# ---------------------------------------------------------------------------

def compute_svd(weight: torch.Tensor):
    """Compute full SVD of a weight matrix. Returns U, S, Vt."""
    # Work in float32 for numerical stability
    W = weight.detach().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    return U, S, Vt


def apply_svd_scaling(weight: torch.Tensor, U, S, Vt, direction_idx: int,
                      scale_factor: float):
    """
    Apply a rank-1 modification to a weight matrix by scaling a single
    singular direction. Modifies the weight tensor in-place and returns
    the delta so it can be undone.

    W' = W + (scale - 1) * sigma_i * u_i @ v_i^T
    """
    sigma = S[direction_idx].to(weight.dtype).to(weight.device)
    u = U[:, direction_idx].to(weight.dtype).to(weight.device)
    v = Vt[direction_idx, :].to(weight.dtype).to(weight.device)

    delta = (scale_factor - 1.0) * sigma * torch.outer(u, v)
    weight.data.add_(delta)
    return delta


def undo_svd_scaling(weight: torch.Tensor, delta: torch.Tensor):
    """Undo a previously applied SVD scaling."""
    weight.data.sub_(delta)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def extract_final_number(text: str) -> str | None:
    """Extract the final numerical answer from model output or GSM8K gold."""
    # GSM8K gold answers use #### <number>
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    # Otherwise grab the last number in the text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def evaluate_math(model, tokenizer, problems, max_new_tokens=512):
    """
    Evaluate the model on a list of GSM8K problems.
    Returns (num_correct, num_total, list_of_results).
    """
    correct = 0
    results = []

    for prob in problems:
        question = prob["question"]
        gold_answer = extract_final_number(prob["answer"])

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False  # use non-thinking mode for speed
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for deterministic comparison
                temperature=None,
                top_p=None,
            )

        # Decode only the generated part
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred_answer = extract_final_number(response)
        is_correct = (pred_answer is not None and gold_answer is not None
                      and pred_answer == gold_answer)
        if is_correct:
            correct += 1

        results.append({
            "question": question[:80] + "...",
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct,
        })

    return correct, len(problems), results


# ---------------------------------------------------------------------------
# Main exploration loop
# ---------------------------------------------------------------------------

def get_weight_tensor(model, layer_idx: int, matrix_name: str):
    """Get a reference to a specific weight tensor in the model."""
    path = f"model.layers.{layer_idx}.{matrix_name}.weight"
    parts = path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def select_layers(model, requested_layers=None):
    """Select which layers to explore. Defaults to a spread across the model."""
    num_layers = len(model.model.layers)
    if requested_layers is not None:
        return [l for l in requested_layers if l < num_layers]
    # Pick early, middle, and late layers
    if num_layers <= 6:
        return list(range(num_layers))
    return [0, num_layers // 4, num_layers // 2,
            3 * num_layers // 4, num_layers - 1]


def run_exploration(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")

    # Load GSM8K problems
    print(f"Loading {args.num_problems} GSM8K problems...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    # Take a fixed subset for consistent comparison
    problems = list(dataset.select(range(min(args.num_problems, len(dataset)))))

    # Determine which layers and matrices to explore
    layers = select_layers(model, args.layers)
    matrices = args.matrices
    print(f"Layers to explore: {layers}")
    print(f"Matrices to explore: {matrices}")
    print(f"Directions per matrix: {args.num_directions}")
    print(f"Scaling factors: {SCALING_FACTORS}")
    print()

    # --- Baseline evaluation ---
    print("=" * 70)
    print("BASELINE EVALUATION (no SVD modifications)")
    print("=" * 70)
    t0 = time.time()
    base_correct, base_total, base_results = evaluate_math(
        model, tokenizer, problems, args.max_new_tokens
    )
    base_acc = base_correct / base_total
    elapsed = time.time() - t0
    print(f"Baseline: {base_correct}/{base_total} = {base_acc:.1%} "
          f"({elapsed:.1f}s)")
    print()

    # --- Precompute SVDs ---
    print("Precomputing SVDs...")
    svd_cache = {}  # (layer, matrix) -> (U, S, Vt)
    for layer_idx in layers:
        for matrix_name in matrices:
            try:
                weight = get_weight_tensor(model, layer_idx, matrix_name)
                U, S, Vt = compute_svd(weight)
                svd_cache[(layer_idx, matrix_name)] = (U, S, Vt)
                # Print the spectrum for intuition
                top_s = S[:args.num_directions].tolist()
                top_s_str = ", ".join(f"{s:.1f}" for s in top_s)
                print(f"  layer {layer_idx:2d} / {matrix_name:20s}: "
                      f"top singular values = [{top_s_str}]")
            except (AttributeError, IndexError) as e:
                print(f"  layer {layer_idx:2d} / {matrix_name:20s}: "
                      f"SKIPPED ({e})")
    print()

    # --- Sweep ---
    all_results = []
    total_configs = (len(svd_cache) * args.num_directions
                     * (len(SCALING_FACTORS) - 1))  # skip 1.0
    print("=" * 70)
    print(f"SWEEPING {total_configs} CONFIGURATIONS")
    print("=" * 70)

    config_idx = 0
    for (layer_idx, matrix_name), (U, S, Vt) in svd_cache.items():
        weight = get_weight_tensor(model, layer_idx, matrix_name)
        max_dir = min(args.num_directions, len(S))

        for dir_idx in range(max_dir):
            for scale in SCALING_FACTORS:
                if scale == 1.0:
                    continue  # that's the baseline

                config_idx += 1
                label = (f"[{config_idx}/{total_configs}] "
                         f"layer={layer_idx} matrix={matrix_name} "
                         f"dir={dir_idx} scale={scale}")

                # Apply modification
                delta = apply_svd_scaling(weight, U, S, Vt, dir_idx, scale)

                # Evaluate
                t0 = time.time()
                correct, total, _ = evaluate_math(
                    model, tokenizer, problems, args.max_new_tokens
                )
                elapsed = time.time() - t0
                acc = correct / total
                diff = acc - base_acc

                # Undo modification
                undo_svd_scaling(weight, delta)

                # Report
                marker = ""
                if diff > 0.05:
                    marker = " *** IMPROVEMENT ***"
                elif diff < -0.1:
                    marker = " (degraded)"

                print(f"{label}: {correct}/{total} = {acc:.1%} "
                      f"(delta={diff:+.1%}, {elapsed:.1f}s){marker}")

                all_results.append({
                    "layer": layer_idx,
                    "matrix": matrix_name,
                    "direction": dir_idx,
                    "singular_value": S[dir_idx].item(),
                    "scale_factor": scale,
                    "correct": correct,
                    "total": total,
                    "accuracy": acc,
                    "delta_vs_baseline": diff,
                })

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline accuracy: {base_acc:.1%}")
    print()

    if all_results:
        # Sort by delta
        sorted_results = sorted(all_results, key=lambda r: r["delta_vs_baseline"],
                                reverse=True)

        print("Top 10 improvements:")
        for r in sorted_results[:10]:
            print(f"  {r['delta_vs_baseline']:+.1%}  layer={r['layer']} "
                  f"{r['matrix']} dir={r['direction']} "
                  f"scale={r['scale_factor']} "
                  f"(sigma={r['singular_value']:.1f})")

        print()
        print("Top 10 degradations:")
        for r in sorted_results[-10:]:
            print(f"  {r['delta_vs_baseline']:+.1%}  layer={r['layer']} "
                  f"{r['matrix']} dir={r['direction']} "
                  f"scale={r['scale_factor']} "
                  f"(sigma={r['singular_value']:.1f})")

        # Save full results
        output_path = Path("svd_exploration_results.json")
        with open(output_path, "w") as f:
            json.dump({
                "model": args.model,
                "baseline_accuracy": base_acc,
                "num_problems": args.num_problems,
                "results": all_results,
            }, f, indent=2)
        print(f"\nFull results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Brute-force SVD direction scaling exploration"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num_problems", type=int,
                        default=DEFAULT_NUM_PROBLEMS,
                        help="Number of GSM8K problems to evaluate per config")
    parser.add_argument("--num_directions", type=int,
                        default=DEFAULT_NUM_DIRECTIONS,
                        help="Number of top singular directions to try")
    parser.add_argument("--max_new_tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to explore (default: auto)")
    parser.add_argument("--matrices", type=str, nargs="+",
                        default=DEFAULT_MATRICES,
                        help="Weight matrix names to explore")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_exploration(args)
