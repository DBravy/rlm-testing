"""
Capital Retrieval: SVD Configuration + LoRA Co-Adaptation

Phase 1: Velocity analysis across all countries to find the best universal
         SVD configuration (the one that moves the most countries toward
         their correct capital).

Phase 2: Apply that SVD config, then LoRA fine-tune on capital completions.
         The LoRA weights learn to exploit the modified weight geometry.

Phase 3: Evaluate all four conditions:
         - Baseline (no SVD, no LoRA)
         - SVD only
         - LoRA only
         - SVD + LoRA

    python train_capital.py
    python train_capital.py --num_candidates 20
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from capital_svd import (
    SVDManager, VelocityAnalyzer, Scaffold,
    split_dataset, build_capital_prompt, config_to_text,
    CAPITALS,
)


# ---------------------------------------------------------------------------
# Phase 1: Find best universal SVD configuration
# ---------------------------------------------------------------------------

def analyze_all_countries(analyzer, countries):
    """Run velocity analysis on every country. Returns list of analyses."""
    analyses = []
    for country, capital in countries:
        prompt = build_capital_prompt(country)
        a = analyzer.analyze(prompt, capital)
        a["country"] = country
        a["capital"] = capital
        analyses.append(a)
    return analyses


def rank_components(analyses):
    """
    Across all countries, rank components by how consistently they
    widen the gap (positive gap contribution = bad).

    Returns a sorted list of (layer, type, matrix, avg_gap, count).
    """
    # Aggregate gap contributions per component
    component_gaps = defaultdict(list)

    for a in analyses:
        all_layers = [lr["layer"] for lr in a["layer_results"]]
        final_layer = max(all_layers) if all_layers else 0

        for lr in a["layer_results"]:
            if lr["layer"] >= final_layer:
                continue
            key_attn = (lr["layer"], "attn", "attn.c_proj")
            key_mlp_down = (lr["layer"], "mlp_down", "mlp.c_proj")
            key_mlp_up = (lr["layer"], "mlp_up", "mlp.c_fc")
            component_gaps[key_attn].append(lr["attn_gap"])
            component_gaps[key_mlp_down].append(lr["mlp_gap"])
            component_gaps[key_mlp_up].append(lr["mlp_gap"])

        for hr in a["head_results"]:
            if hr["layer"] >= final_layer:
                continue
            key = (hr["layer"], f"head_{hr['head']}", "attn.c_proj")
            component_gaps[key].append(hr["gap"])

    # Compute average gap per component
    ranked = []
    for (layer, comp_type, matrix), gaps in component_gaps.items():
        avg_gap = sum(gaps) / len(gaps)
        ranked.append({
            "layer": layer, "type": comp_type, "matrix": matrix,
            "avg_gap": avg_gap, "count": len(gaps),
        })

    ranked.sort(key=lambda x: x["avg_gap"], reverse=True)
    return ranked


def build_candidate_configs(ranked_components, directions=(0, 1, 2),
                            suppress_scales=(0.0, 0.3, 0.5),
                            amplify_scales=(1.5, 2.0, 3.0),
                            max_candidates=30):
    """
    Build candidate universal configs from the ranked component analysis.

    Strategy: suppress the worst offenders, amplify the best helpers,
    and try combinations.
    """
    configs = []

    # Worst offenders (largest positive avg gap)
    worst = [c for c in ranked_components if c["avg_gap"] > 0][:5]
    # Best helpers (most negative avg gap)
    best = [c for c in ranked_components if c["avg_gap"] < 0]
    best.sort(key=lambda c: c["avg_gap"])
    best = best[:5]

    # Single-component suppression
    for comp in worst:
        for d in directions:
            for s in suppress_scales:
                configs.append([(comp["layer"], comp["matrix"], d, s)])

    # Single-component amplification
    for comp in best:
        for d in directions:
            for s in amplify_scales:
                configs.append([(comp["layer"], comp["matrix"], d, s)])

    # Combos: suppress worst + amplify best
    for w in worst[:3]:
        for b in best[:3]:
            if w["layer"] == b["layer"] and w["matrix"] == b["matrix"]:
                continue
            for wd in directions[:2]:
                for bd in directions[:2]:
                    configs.append([
                        (w["layer"], w["matrix"], wd, 0.0),
                        (b["layer"], b["matrix"], bd, 2.0),
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
    return unique[:max_candidates]


def get_target_rank(model, tokenizer, prompt, target_token):
    """Get the rank of target_token in the model's predictions for prompt."""
    # Get target token id
    ids_with_space = tokenizer.encode(" " + target_token)
    ids_without = tokenizer.encode(target_token)
    if len(ids_with_space) == 1:
        target_id = ids_with_space[0]
    elif len(ids_without) == 1:
        target_id = ids_without[0]
    else:
        target_id = ids_with_space[0]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].float()
    sorted_indices = logits.argsort(descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item()
    return rank


def compute_baseline_ranks(model, tokenizer, countries):
    """Compute baseline rank of correct capital for every country. Done once."""
    ranks = {}
    for country, capital in countries:
        prompt = build_capital_prompt(country)
        rank = get_target_rank(model, tokenizer, prompt, capital)
        ranks[country] = rank
    return ranks


def score_config(model, tokenizer, svd_manager, config, countries,
                 baseline_ranks):
    """
    Score a config by how much it improves the rank of correct answers.

    Returns:
      - in_top5: how many countries have correct token in top 5 after config
      - avg_rank: average rank of correct token after config (lower = better)
      - avg_improvement: average rank improvement vs baseline (positive = better)
      - per_country: list of (country, baseline_rank, config_rank)
    """
    svd_manager.reset_all()
    svd_manager.apply_config(config)

    in_top5 = 0
    total_rank = 0
    total_improvement = 0
    per_country = []

    for country, capital in countries:
        prompt = build_capital_prompt(country)
        config_rank = get_target_rank(model, tokenizer, prompt, capital)
        base_rank = baseline_ranks[country]
        improvement = base_rank - config_rank  # positive = rank got better

        if config_rank < 5:
            in_top5 += 1
        total_rank += config_rank
        total_improvement += improvement
        per_country.append((country, base_rank, config_rank))

    svd_manager.reset_all()

    n = len(countries)
    return {
        "in_top5": in_top5,
        "avg_rank": total_rank / n if n > 0 else 999,
        "avg_improvement": total_improvement / n if n > 0 else 0,
        "per_country": per_country,
    }


def find_best_config(model, tokenizer, scaffold, svd_manager, analyzer,
                     train_set, num_candidates=30):
    """
    Phase 1: Analyze all countries, rank components, build candidate
    configs, score by rank improvement, return the best.
    """
    print("Phase 1: Analyzing all training countries...")
    analyses = analyze_all_countries(analyzer, train_set)

    already_correct = sum(1 for a in analyses if a["already_correct"])
    print(f"  {len(analyses)} countries analyzed")
    print(f"  {already_correct} already correct at baseline")
    print(f"  {len(analyses) - already_correct} need help")

    # Compute baseline ranks (done once, reused for all configs)
    print("\nComputing baseline ranks...")
    baseline_ranks = compute_baseline_ranks(model, tokenizer, train_set)
    ranks_list = list(baseline_ranks.values())
    avg_base = sum(ranks_list) / len(ranks_list) if ranks_list else 0
    in_top5_base = sum(1 for r in ranks_list if r < 5)
    print(f"  Baseline avg rank: {avg_base:.1f}")
    print(f"  Baseline in top 5: {in_top5_base}/{len(train_set)}")

    # Rank components
    print("\nRanking components by average gap contribution...")
    ranked = rank_components(analyses)
    print("  Top 10 worst offenders (avg gap across countries):")
    for c in ranked[:10]:
        print(f"    L{c['layer']}/{c['type']:>10s} ({c['matrix']:>12s}): "
              f"avg_gap={c['avg_gap']:>+.2f}")
    print("  Top 5 best helpers:")
    for c in ranked[-5:]:
        print(f"    L{c['layer']}/{c['type']:>10s} ({c['matrix']:>12s}): "
              f"avg_gap={c['avg_gap']:>+.2f}")

    # Build and score candidates
    candidates = build_candidate_configs(ranked, max_candidates=num_candidates)
    print(f"\nScoring {len(candidates)} candidate configs...")

    best_config = None
    best_in_top5 = -1
    best_avg_rank = 9999

    for i, cfg in enumerate(candidates):
        score = score_config(model, tokenizer, svd_manager, cfg,
                             train_set, baseline_ranks)
        cfg_str = config_to_text(cfg)
        if len(cfg_str) > 50:
            cfg_str = cfg_str[:47] + "..."
        print(f"  [{i+1:>2}/{len(candidates)}] "
              f"top5={score['in_top5']:>2d} "
              f"avg_rank={score['avg_rank']:>6.1f} "
              f"avg_impr={score['avg_improvement']:>+6.1f} | "
              f"{cfg_str}")

        # Best = most in top 5, tiebreak by avg rank
        if (score["in_top5"] > best_in_top5 or
            (score["in_top5"] == best_in_top5 and
             score["avg_rank"] < best_avg_rank)):
            best_in_top5 = score["in_top5"]
            best_avg_rank = score["avg_rank"]
            best_config = cfg

    print(f"\nBest config: top5={best_in_top5}, avg_rank={best_avg_rank:.1f}")
    print(f"  {config_to_text(best_config)}")
    print(f"  (baseline was: top5={in_top5_base}, avg_rank={avg_base:.1f})")
    return best_config


# ---------------------------------------------------------------------------
# Phase 2: LoRA fine-tune with SVD config applied
# ---------------------------------------------------------------------------

def build_capital_training_data(tokenizer, countries):
    """Build simple completion training examples: 'The capital of X is Y'"""
    texts = []
    for country, capital in countries:
        # Use the full capital name from our list
        texts.append(f"The capital of {country} is {capital}")
    return texts


def lora_finetune(model, tokenizer, train_texts, output_dir, lr=5e-5,
                  epochs=10, batch_size=4):
    """Run LoRA fine-tuning on the capital completion examples."""
    print(f"  Fine-tuning on {len(train_texts)} examples for {epochs} epochs...")

    encodings = tokenizer(
        train_texts, truncation=True, max_length=64, padding=True,
        return_tensors="pt",
    )

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return self.enc["input_ids"].shape[0]
        def __getitem__(self, idx):
            return {"input_ids": self.enc["input_ids"][idx],
                    "attention_mask": self.enc["attention_mask"][idx],
                    "labels": self.enc["input_ids"][idx].clone()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            fp16=False,
        ),
        train_dataset=DS(encodings),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    print("  Fine-tuning complete.")


# ---------------------------------------------------------------------------
# Phase 3: Evaluate all four conditions
# ---------------------------------------------------------------------------

def evaluate_condition(scaffold, svd_manager, countries, config=None,
                       label=""):
    """Evaluate accuracy with optional SVD config applied."""
    if config:
        svd_manager.reset_all()
        svd_manager.apply_config(config)

    correct = 0
    results = []
    for country, capital in countries:
        pred = scaffold._get_top_token(build_capital_prompt(country))
        ok = scaffold._check_capital(pred, capital)
        if ok:
            correct += 1
        results.append((country, capital, pred, ok))

    svd_manager.reset_all()

    n = len(countries)
    print(f"  {label}: {correct}/{n} ({correct/n:.1%})")
    return correct, n, results


def full_evaluation(model, tokenizer, svd_manager, eval_set, best_config,
                    lora_adapter_path=None):
    """
    Evaluate all four conditions:
    1. Baseline (no SVD, no LoRA)
    2. SVD only
    3. LoRA only
    4. SVD + LoRA
    """
    print("\n" + "=" * 60)
    print("FULL EVALUATION (4 conditions)")
    print("=" * 60)

    scaffold = Scaffold(model, tokenizer, svd_manager)

    # Condition 1: Baseline (model should already have LoRA merged or not)
    # We need to handle LoRA enable/disable carefully
    # If the model is a PeftModel, we can disable adapters
    is_peft = hasattr(model, "disable_adapter_layers")

    # Condition 1: No SVD, No LoRA
    if is_peft:
        model.disable_adapter_layers()
    c1_correct, c1_n, c1_results = evaluate_condition(
        scaffold, svd_manager, eval_set, config=None,
        label="Baseline (no SVD, no LoRA)")

    # Condition 2: SVD only, No LoRA
    if is_peft:
        model.disable_adapter_layers()
    c2_correct, c2_n, c2_results = evaluate_condition(
        scaffold, svd_manager, eval_set, config=best_config,
        label="SVD only")

    # Condition 3: LoRA only, No SVD
    if is_peft:
        model.enable_adapter_layers()
    c3_correct, c3_n, c3_results = evaluate_condition(
        scaffold, svd_manager, eval_set, config=None,
        label="LoRA only")

    # Condition 4: SVD + LoRA
    if is_peft:
        model.enable_adapter_layers()
    c4_correct, c4_n, c4_results = evaluate_condition(
        scaffold, svd_manager, eval_set, config=best_config,
        label="SVD + LoRA")

    # Per-country comparison for interesting cases
    print(f"\n  Per-country breakdown (training accuracy):")
    print(f"  {'Country':<20s} | {'Base':>5s} {'SVD':>5s} {'LoRA':>5s} {'Both':>5s} | Gold")
    print(f"  {'-'*20}-+-{'-'*5}-{'-'*5}-{'-'*5}-{'-'*5}-+------")
    for i in range(len(eval_set)):
        country = eval_set[i][0]
        capital = eval_set[i][1]
        b = c1_results[i][2]
        s = c2_results[i][2]
        l = c3_results[i][2]
        sl = c4_results[i][2]
        b_ok = "Y" if c1_results[i][3] else ""
        s_ok = "Y" if c2_results[i][3] else ""
        l_ok = "Y" if c3_results[i][3] else ""
        sl_ok = "Y" if c4_results[i][3] else ""
        print(f"  {country:<20s} | {b_ok:>5s} {s_ok:>5s} {l_ok:>5s} {sl_ok:>5s} | {capital}")

    return {
        "baseline": c1_correct / c1_n,
        "svd_only": c2_correct / c2_n,
        "lora_only": c3_correct / c3_n,
        "svd_lora": c4_correct / c4_n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    print("Building SVD manager (on base model, before LoRA)...")
    svd_manager = SVDManager(model, max_directions=args.max_directions)
    analyzer = VelocityAnalyzer(model, tokenizer)
    scaffold = Scaffold(model, tokenizer, svd_manager)

    # Use ALL capitals for both training and evaluation
    # (capital retrieval is memorization, not generalization)
    all_countries = list(CAPITALS)
    print(f"Total countries: {len(all_countries)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Phase 1: Find best SVD config
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: FIND BEST SVD CONFIGURATION")
    print(f"{'=' * 60}")

    best_config = find_best_config(
        model, tokenizer, scaffold, svd_manager, analyzer, all_countries,
        num_candidates=args.num_candidates,
    )

    # Save config
    config_path = output_dir / "best_config.json"
    with open(config_path, "w") as f:
        json.dump({"config": best_config}, f, indent=2)
    print(f"Config saved to {config_path}")

    # Quick eval of SVD-only
    print("\nSVD-only eval:")
    evaluate_condition(scaffold, svd_manager, all_countries, config=best_config,
                       label="SVD only")

    # =====================================================================
    # Phase 2: LoRA fine-tune with SVD config applied
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2: LoRA FINE-TUNE WITH SVD CONFIG APPLIED")
    print(f"{'=' * 60}")

    # Apply SVD config BEFORE creating LoRA
    print("Applying SVD config to base weights...")
    svd_manager.reset_all()
    svd_manager.apply_config(best_config)

    # Now apply LoRA on top of the SVD-modified model
    print("Adding LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Update scaffold to use LoRA model
    scaffold = Scaffold(model, tokenizer, svd_manager)

    # Build training data
    train_texts = build_capital_training_data(tokenizer, all_countries)
    print(f"Training examples: {len(train_texts)}")
    print(f"  Example: '{train_texts[0]}'")

    # Fine-tune
    lora_finetune(model, tokenizer, train_texts,
                  str(output_dir / "lora"),
                  lr=args.learning_rate, epochs=args.epochs)

    # Undo SVD config (LoRA weights were trained WITH it applied,
    # but we need clean base weights to test conditions independently)
    svd_manager.reset_all()

    # Save LoRA adapter
    lora_path = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_path))
    print(f"LoRA adapter saved to {lora_path}")

    # =====================================================================
    # Phase 3: Evaluate all four conditions
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 3: EVALUATE ALL CONDITIONS")
    print(f"{'=' * 60}")

    results = full_evaluation(
        model, tokenizer, svd_manager, all_countries, best_config
    )

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:   {results['baseline']:.1%}")
    print(f"  SVD only:   {results['svd_only']:.1%}")
    print(f"  LoRA only:  {results['lora_only']:.1%}")
    print(f"  SVD + LoRA: {results['svd_lora']:.1%}")

    improvement = results["svd_lora"] - results["baseline"]
    lora_contribution = results["lora_only"] - results["baseline"]
    svd_contribution = results["svd_only"] - results["baseline"]
    synergy = improvement - lora_contribution - svd_contribution
    print(f"\n  Total improvement:   {improvement:+.1%}")
    print(f"  LoRA contribution:   {lora_contribution:+.1%}")
    print(f"  SVD contribution:    {svd_contribution:+.1%}")
    print(f"  Synergy (co-adapt):  {synergy:+.1%}")

    if synergy > 0:
        print("\n  >>> Positive synergy: SVD + LoRA together outperform "
              "the sum of their individual contributions!")
    elif synergy < -0.01:
        print("\n  --- Negative synergy: SVD + LoRA interfere with each other")
    else:
        print("\n  Neutral synergy: contributions are roughly additive")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_config": best_config,
            "results": results,
            "synergy": synergy,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--num_candidates", type=int, default=20,
                        help="SVD config candidates to score in Phase 1")
    parser.add_argument("--max_directions", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", default="capital_checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
