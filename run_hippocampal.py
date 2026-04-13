"""
Hippocampal LoRA Experiment Runner

Phase 1: Train separate LoRA adapters for each knowledge domain
Phase 2: Collect bottleneck signatures, store in Hopfield attractor
Phase 3: Test recall on domain-specific prompts (does it pick the right adapter?)
Phase 4: Test ambiguous prompts (does blending produce something useful?)
Phase 5: Apply recalled adapters and check if they actually help

    python run_hippocampal.py
    python run_hippocampal.py --domains capitals elements languages
    python run_hippocampal.py --beta 12.0 --epochs 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hippocampal_lora import (
    DOMAINS, AMBIGUOUS_PROMPTS,
    train_domain_adapter, evaluate_domain,
    extract_lora_A_matrices, extract_lora_B_matrices,
    collect_bottleneck_activations, activations_to_signature,
    HopfieldAttractor, DomainAdapter, AdapterMemory,
    apply_adapter_matrices, get_cue_for_prompt,
)


def main(args):
    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model)

    domains = args.domains
    print(f"Domains: {domains}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Phase 1: Train adapters
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: TRAIN DOMAIN-SPECIFIC LoRA ADAPTERS")
    print(f"{'=' * 60}")

    trained_models = {}
    train_accuracies = {}

    for domain in domains:
        print(f"\n--- Training: {domain} ({len(DOMAINS[domain])} facts) ---")
        t0 = time.time()
        peft_model = train_domain_adapter(
            base_model, tokenizer, domain,
            lr=args.learning_rate, epochs=args.epochs,
            rank=args.rank, output_dir=str(output_dir / "adapters"),
        )
        elapsed = time.time() - t0

        # Evaluate on training data
        correct, total = evaluate_domain(peft_model, tokenizer, domain)
        acc = correct / total
        train_accuracies[domain] = acc
        print(f"  {domain}: {correct}/{total} ({acc:.1%}) in {elapsed:.1f}s")

        trained_models[domain] = peft_model

    # Cross-domain evaluation: how does each adapter do on OTHER domains?
    print(f"\n--- Cross-domain evaluation ---")
    print(f"  {'':>12s}", end="")
    for d in domains:
        print(f" {d:>10s}", end="")
    print()

    for adapter_domain in domains:
        print(f"  {adapter_domain:>12s}", end="")
        for eval_domain in domains:
            correct, total = evaluate_domain(
                trained_models[adapter_domain], tokenizer, eval_domain
            )
            pct = correct / total
            marker = " *" if adapter_domain == eval_domain else ""
            print(f" {pct:>9.1%}{marker}", end="")
        print()

    # =====================================================================
    # Phase 2: Collect signatures, build attractor
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2: COLLECT BOTTLENECK SIGNATURES")
    print(f"{'=' * 60}")

    memory = AdapterMemory(beta=args.beta)

    for domain in domains:
        peft_model = trained_models[domain]

        # Extract A and B matrices
        A_mats = extract_lora_A_matrices(peft_model)
        B_mats = extract_lora_B_matrices(peft_model)

        # Collect bottleneck activations on domain prompts
        prompts = [prompt for prompt, _ in DOMAINS[domain]]
        activations = collect_bottleneck_activations(
            peft_model, tokenizer, prompts, A_mats
        )
        signature = activations_to_signature(activations)

        adapter = DomainAdapter(
            name=domain,
            A_matrices=A_mats,
            B_matrices=B_mats,
            signature=signature,
            train_accuracy=train_accuracies[domain],
        )
        memory.register_adapter(adapter)

        print(f"  {domain}: signature dim={signature.shape[0]}, "
              f"norm={torch.linalg.norm(signature):.3f}")

    # Print signature similarity matrix
    print(f"\n  Signature similarity (cosine):")
    print(f"  {'':>12s}", end="")
    for d in domains:
        print(f" {d:>10s}", end="")
    print()

    for d1 in domains:
        print(f"  {d1:>12s}", end="")
        s1 = memory.adapters[d1].signature
        for d2 in domains:
            s2 = memory.adapters[d2].signature
            sim = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
            print(f" {sim:>10.3f}", end="")
        print()

    # =====================================================================
    # Phase 3: Test domain recall on domain-specific prompts
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 3: TEST DOMAIN RECALL")
    print(f"{'=' * 60}")

    # For each domain, take a few prompts, compute their cue, and see
    # if the attractor recalls the right adapter

    # We need a reference set of A matrices to compute cues.
    # Use the first domain's A matrices as the "probe" encoder.
    # In a full system this would be a separate learned projection.
    ref_A = memory.adapters[domains[0]].A_matrices

    recall_results = {}

    for domain in domains:
        test_prompts = [prompt for prompt, _ in DOMAINS[domain][:5]]
        correct_recall = 0

        for prompt in test_prompts:
            cue = get_cue_for_prompt(
                trained_models[domains[0]], tokenizer, prompt, ref_A
            )
            if cue is None:
                continue

            top_domain, scores = memory.attractor.retrieve_top(cue)

            is_correct = (top_domain == domain)
            if is_correct:
                correct_recall += 1

            # Print first prompt per domain with full scores
            if prompt == test_prompts[0]:
                scores_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in
                    sorted(scores.items(), key=lambda x: -x[1])
                )
                status = "CORRECT" if is_correct else f"WRONG (got {top_domain})"
                print(f"  [{domain}] '{prompt[:50]}...'")
                print(f"    Recalled: {status}")
                print(f"    Scores: {scores_str}")

        recall_acc = correct_recall / len(test_prompts) if test_prompts else 0
        recall_results[domain] = recall_acc
        print(f"  {domain} recall: {correct_recall}/{len(test_prompts)} "
              f"({recall_acc:.1%})")

    # =====================================================================
    # Phase 4: Test ambiguous prompts
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 4: AMBIGUOUS PROMPTS")
    print(f"{'=' * 60}")

    for prompt_text, expected_domains in AMBIGUOUS_PROMPTS:
        # Only test if at least one expected domain is in our set
        relevant = [d for d in expected_domains if d in domains]
        if not relevant:
            continue

        cue = get_cue_for_prompt(
            trained_models[domains[0]], tokenizer, prompt_text, ref_A
        )
        if cue is None:
            continue

        _, _, scores = memory.attractor.retrieve(cue)
        scores_str = ", ".join(
            f"{k}={v:.3f}" for k, v in
            sorted(scores.items(), key=lambda x: -x[1])
        )
        print(f"\n  '{prompt_text}'")
        print(f"    Expected domains: {relevant}")
        print(f"    Attractor scores: {scores_str}")

        # Is the top score one of the expected domains?
        top = max(scores, key=scores.get)
        if top in relevant:
            print(f"    Top recall ({top}) matches expected domain")
        else:
            print(f"    Top recall ({top}) NOT in expected domains")

    # =====================================================================
    # Phase 5: Apply recalled adapter and verify it helps
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 5: APPLY RECALLED ADAPTERS")
    print(f"{'=' * 60}")

    # For each domain, test: does applying the attractor-recalled adapter
    # actually improve performance vs baseline and vs wrong adapter?

    # First, create a fresh peft model with empty LoRA to overwrite
    from peft import LoraConfig, get_peft_model, TaskType
    import copy

    eval_model = copy.deepcopy(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    eval_model = get_peft_model(eval_model, lora_config)

    for domain in domains:
        # Get the correct adapter for this domain
        correct_adapter = memory.adapters[domain]

        # Get a wrong adapter (first one that isn't this domain)
        wrong_domain = [d for d in domains if d != domain][0]
        wrong_adapter = memory.adapters[wrong_domain]

        # Test with correct adapter
        apply_adapter_matrices(
            eval_model, correct_adapter.A_matrices, correct_adapter.B_matrices
        )
        correct_score, total = evaluate_domain(eval_model, tokenizer, domain)

        # Test with wrong adapter
        apply_adapter_matrices(
            eval_model, wrong_adapter.A_matrices, wrong_adapter.B_matrices
        )
        wrong_score, _ = evaluate_domain(eval_model, tokenizer, domain)

        # Test with blended adapter from attractor
        # Get a cue from the first prompt in this domain
        first_prompt = DOMAINS[domain][0][0]
        cue = get_cue_for_prompt(
            trained_models[domains[0]], tokenizer, first_prompt, ref_A
        )
        if cue is not None:
            blended_A, blended_B, blend_scores = memory.recall_blended_matrices(
                cue
            )
            if blended_A is not None:
                apply_adapter_matrices(eval_model, blended_A, blended_B)
                blended_score, _ = evaluate_domain(eval_model, tokenizer, domain)
            else:
                blended_score = 0
        else:
            blended_score = 0
            blend_scores = {}

        print(f"\n  {domain}:")
        print(f"    Correct adapter:  {correct_score}/{total} "
              f"({correct_score/total:.1%})")
        print(f"    Wrong adapter:    {wrong_score}/{total} "
              f"({wrong_score/total:.1%})")
        print(f"    Blended adapter:  {blended_score}/{total} "
              f"({blended_score/total:.1%})")
        if blend_scores:
            bs = ", ".join(f"{k}={v:.2f}" for k, v in
                           sorted(blend_scores.items(), key=lambda x: -x[1]))
            print(f"    Blend weights: {bs}")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nTraining accuracy per domain:")
    for domain in domains:
        print(f"  {domain:>12s}: {train_accuracies[domain]:.1%}")
    print(f"\nAttractor recall accuracy (does it pick the right adapter?):")
    for domain in domains:
        print(f"  {domain:>12s}: {recall_results.get(domain, 0):.1%}")

    # Save results
    results = {
        "domains": domains,
        "train_accuracies": train_accuracies,
        "recall_results": recall_results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--domains", nargs="+",
                        default=["capitals", "elements", "languages",
                                 "opposites", "colors"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--beta", type=float, default=8.0,
                        help="Hopfield inverse temperature (higher = sharper)")
    parser.add_argument("--output_dir", default="hippocampal_checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
