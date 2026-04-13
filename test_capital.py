"""
Test capital retrieval with competitive velocity analysis.

    python test_capital.py
    python test_capital.py --country France
    python test_capital.py --num_countries 20
"""

import argparse
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from capital_svd import (
    SVDManager, VelocityAnalyzer, Scaffold,
    split_dataset, build_capital_prompt, config_to_text,
    generate_targeted_configs,
)


def print_analysis(analysis):
    """Pretty-print the competitive velocity analysis (actionable layers only)."""
    print(f"  Target: '{analysis['target_token']}' "
          f"(logit={analysis['target_logit']:.2f}, rank={analysis['target_rank']})")
    print(f"  Winner: '{analysis['winner_token']}' "
          f"(logit={analysis['winner_logit']:.2f})")
    leverage = analysis.get("leverage_ratio", 0)
    if analysis["already_correct"]:
        triage_str = "ALREADY CORRECT"
    elif analysis["closeable"]:
        triage_str = f"closeable, leverage={leverage:.1f}x"
    else:
        triage_str = f"HARD, leverage={leverage:.1f}x"
    print(f"  Gap: {analysis['gap']:.2f} ({triage_str})")
    print(f"  Top 5: {', '.join(f'{t}({l:.1f})' for t, l in analysis['top5'])}")

    all_layer_indices = [lr["layer"] for lr in analysis["layer_results"]]
    final_layer = max(all_layer_indices) if all_layer_indices else 0

    print(f"\n  Actionable layers (excluding final layer {final_layer}):")
    print(f"  {'Layer':>5} | {'Attn->Tgt':>9} {'Attn->Win':>9} {'Attn Gap':>9} | "
          f"{'MLP->Tgt':>9} {'MLP->Win':>9} {'MLP Gap':>9}")
    print(f"  {'-'*5}-+-{'-'*31}-+-{'-'*31}")

    for lr in analysis["layer_results"]:
        if lr["layer"] >= final_layer:
            continue
        at = lr["attn_to_target"]
        aw = lr["attn_to_winner"]
        ag = lr["attn_gap"]
        mt = lr["mlp_to_target"]
        mw = lr["mlp_to_winner"]
        mg = lr["mlp_gap"]
        a_mark = " !!" if ag > 1.0 else ""
        m_mark = " !!" if mg > 1.0 else ""
        print(f"  {lr['layer']:>5} | {at:>+9.2f} {aw:>+9.2f} {ag:>+9.2f}{a_mark:3s} | "
              f"{mt:>+9.2f} {mw:>+9.2f} {mg:>+9.2f}{m_mark:3s}")

    # Heads (excluding final layer)
    if analysis["head_results"]:
        actionable_heads = [h for h in analysis["head_results"]
                           if h["layer"] < final_layer]
        if actionable_heads:
            heads_by_gap = sorted(actionable_heads,
                                  key=lambda h: h["gap"], reverse=True)
            print(f"\n  Worst gap-widening heads (actionable):")
            for h in heads_by_gap[:5]:
                if h["gap"] > 0:
                    print(f"    L{h['layer']}H{h['head']:>2}: "
                          f"gap={h['gap']:>+.2f} "
                          f"(tgt={h['to_target']:>+.2f}, "
                          f"win={h['to_winner']:>+.2f})")
            print(f"  Best gap-narrowing heads (actionable):")
            for h in heads_by_gap[-5:]:
                if h["gap"] < 0:
                    print(f"    L{h['layer']}H{h['head']:>2}: "
                          f"gap={h['gap']:>+.2f} "
                          f"(tgt={h['to_target']:>+.2f}, "
                          f"win={h['to_winner']:>+.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--country", default=None)
    parser.add_argument("--num_countries", type=int, default=10)
    parser.add_argument("--num_targeted", type=int, default=30)
    parser.add_argument("--max_directions", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    print("Building SVD manager...")
    svd_manager = SVDManager(model, max_directions=args.max_directions)
    analyzer = VelocityAnalyzer(model, tokenizer)
    scaffold = Scaffold(model, tokenizer, svd_manager)

    if args.country:
        from capital_svd import CAPITALS
        match = [(c, cap) for c, cap in CAPITALS
                 if c.lower() == args.country.lower()]
        if not match:
            print(f"Country '{args.country}' not found")
            return
        countries = match
    else:
        _, eval_set = split_dataset()
        countries = eval_set[:args.num_countries]

    baseline_total = 0
    targeted_helped = 0
    skipped_triage = 0

    for country, capital in countries:
        print(f"\n{'=' * 70}")
        print(f"{country} (gold: {capital})")
        print(f"{'=' * 70}")

        base_pred, base_ok = scaffold.baseline_eval(country, capital)
        print(f"\nBaseline: '{base_pred}' [{'CORRECT' if base_ok else 'WRONG'}]")
        if base_ok:
            baseline_total += 1

        prompt = build_capital_prompt(country)
        analysis = analyzer.analyze(prompt, capital)
        print(f"\nCompetitive Velocity Analysis:")
        print_analysis(analysis)

        if analysis["already_correct"]:
            print(f"\n  Already correct, skipping.")
            continue

        if not analysis["closeable"]:
            lev = analysis.get("leverage_ratio", 0)
            print(f"\n  Triage: leverage={lev:.1f}x, not closeable. Skipping.")
            skipped_triage += 1
            continue

        # Generate and test configs
        configs = generate_targeted_configs(
            analysis, svd_manager, num_configs=args.num_targeted
        )
        print(f"\n  Testing {len(configs)} targeted configs:")

        predictions = Counter()
        found = False
        changed_from_baseline = []

        for i, cfg in enumerate(configs):
            episode = scaffold.run_episode(
                country, capital, forced_config=cfg, run_baseline=False
            )
            pred = episode.prediction
            predictions[pred] = predictions.get(pred, 0) + 1

            # Did this config change the prediction from baseline?
            moved = (pred != base_pred)
            correct = episode.correct

            # Print every config with its result
            status = "CORRECT" if correct else "moved" if moved else "same"
            cfg_str = config_to_text(cfg)
            # Truncate long configs for display
            if len(cfg_str) > 60:
                cfg_str = cfg_str[:57] + "..."
            print(f"    [{i+1:>2}] {status:>7} -> '{pred}' | {cfg_str}")

            if moved:
                changed_from_baseline.append((cfg, pred, correct))
            if correct and not found:
                found = True
                targeted_helped += 1

        # Summary for this country
        print(f"\n  Config results for {country}:")
        print(f"    Predictions seen: {dict(predictions)}")
        print(f"    Changed from baseline ('{base_pred}'): "
              f"{len(changed_from_baseline)}/{len(configs)}")
        if found:
            print(f"    >>> FOUND correct config!")
        else:
            print(f"    No config produced '{capital}'")

    n = len(countries)
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Countries tested:    {n}")
    print(f"Baseline correct:    {baseline_total}/{n}")
    print(f"Skipped (triage):    {skipped_triage}")
    print(f"Targeted SVD helped: {targeted_helped} (flipped wrong -> correct)")


if __name__ == "__main__":
    main()
