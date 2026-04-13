"""
Hippocampal LoRA Experiment

Phase 1: Train LoRA adapters per domain. During training, the hippocampus
         observes the model's residual stream states and encodes trajectories
         via Hebbian learning. No gradients flow through the hippocampus.

Phase 2: Test hippocampal recall + activation injection. Given a new prompt,
         the hippocampus pattern-completes from the cue and injects the
         recalled state into the residual stream. Does this improve performance?

Conditions tested:
  - Baseline (no LoRA, no injection)
  - LoRA only (correct adapter loaded)
  - Injection only (hippocampal recall, no LoRA)
  - LoRA + Injection (both active)
  - Wrong LoRA (wrong adapter loaded, as control)

    python run_hippocampal.py
    python run_hippocampal.py --domains capitals elements languages
    python run_hippocampal.py --injection_strength 0.5
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from hippocampal_lora import (
    DOMAINS, domain_to_training_texts,
    HippocampalSystem, ResidualStreamHook,
)


# ---------------------------------------------------------------------------
# Training with hippocampal observation
# ---------------------------------------------------------------------------

def train_domain_with_hippocampus(base_model, tokenizer, domain_name,
                                  hippocampus, hook, lr=5e-5, epochs=15,
                                  rank=16, output_dir="/tmp/lora"):
    """
    Train a LoRA adapter while the hippocampus observes residual stream states.

    The hippocampus encodes each training batch's residual stream as a
    sequential pattern. Learning is purely Hebbian in the hippocampus.
    """
    model = copy.deepcopy(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)

    texts = domain_to_training_texts(domain_name)
    encodings = tokenizer(
        texts, truncation=True, max_length=64, padding=True,
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

    # Train the LoRA adapter (standard gradient-based)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{output_dir}/{domain_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=False,
        ),
        train_dataset=DS(encodings),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    # Now run the trained model on its domain prompts and let the
    # hippocampus observe the residual stream states
    model.eval()
    hook.attach(model, hook.target_layer_idx)
    hippocampus.begin_sequence()

    with torch.no_grad():
        for prompt, answer in DOMAINS[domain_name]:
            hook.clear()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)

            if hook.captured_state is not None:
                hippocampus.encode(hook.captured_state)

    hippocampus.end_sequence()
    hook.detach()

    return model


def evaluate_domain(model, tokenizer, domain_name):
    """Evaluate top-1 accuracy on a domain's facts."""
    correct = 0
    for prompt, answer in DOMAINS[domain_name]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        pred_id = logits.argmax().item()
        pred = tokenizer.decode([pred_id]).strip()
        ok = (pred.lower().startswith(answer.lower()) or
              answer.lower().startswith(pred.lower()))
        if ok:
            correct += 1
    return correct, len(DOMAINS[domain_name])


def evaluate_with_injection(model, tokenizer, domain_name, hippocampus,
                            hook, injection_strength=1.0):
    """
    Evaluate with hippocampal injection active.
    For each prompt: capture residual stream -> hippocampus recall ->
    inject recalled state -> check prediction.
    """
    model.eval()
    correct = 0
    similarities = []

    for prompt, answer in DOMAINS[domain_name]:
        # First pass: capture residual stream state (no injection)
        hook.clear()
        hook.injection = None
        hook.attach(model, hook.target_layer_idx)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        hook.detach()

        if hook.captured_state is None:
            continue

        # Hippocampal recall
        recalled, sim = hippocampus.recall_similarity(hook.captured_state)
        similarities.append(sim)

        # Second pass: with injection
        hook.clear()
        hook.injection = recalled
        hook.injection_strength = injection_strength
        hook.attach(model, hook.target_layer_idx)

        with torch.no_grad():
            outputs = model(**inputs)

        hook.detach()

        logits = outputs.logits[0, -1, :]
        pred_id = logits.argmax().item()
        pred = tokenizer.decode([pred_id]).strip()
        ok = (pred.lower().startswith(answer.lower()) or
              answer.lower().startswith(pred.lower()))
        if ok:
            correct += 1

    avg_sim = sum(similarities) / len(similarities) if similarities else 0
    return correct, len(DOMAINS[domain_name]), avg_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model)

    n_layers = len(base_model.transformer.h)
    d_model = base_model.config.n_embd
    target_layer = args.target_layer if args.target_layer >= 0 else n_layers + args.target_layer

    print(f"Model: {d_model}d, {n_layers} layers")
    print(f"Target layer for residual stream: {target_layer}")
    print(f"Domains: {args.domains}")
    print(f"Injection strength: {args.injection_strength}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize hippocampal system
    print(f"\nInitializing hippocampal system (d_cortex={d_model})...")
    hippocampus = HippocampalSystem(
        d_cortex=d_model,
        d_ec=args.d_ec,
        k_ca3=args.k_ca3,
        ca3_lr=args.ca3_lr,
        ca1_lr=args.ca1_lr,
    )
    hook = ResidualStreamHook(target_layer_idx=target_layer)

    # =====================================================================
    # Phase 1: Train adapters while hippocampus observes
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: TRAIN ADAPTERS + HIPPOCAMPAL ENCODING")
    print(f"{'=' * 60}")

    trained_models = {}

    for domain in args.domains:
        print(f"\n--- {domain} ({len(DOMAINS[domain])} facts) ---")
        t0 = time.time()
        peft_model = train_domain_with_hippocampus(
            base_model, tokenizer, domain, hippocampus, hook,
            lr=args.learning_rate, epochs=args.epochs, rank=args.rank,
            output_dir=str(output_dir / "adapters"),
        )
        elapsed = time.time() - t0

        correct, total = evaluate_domain(peft_model, tokenizer, domain)
        print(f"  LoRA accuracy: {correct}/{total} ({correct/total:.1%}) "
              f"[{elapsed:.1f}s]")
        trained_models[domain] = peft_model

    print(f"\nHippocampus stats: {hippocampus.n_encoded} states encoded, "
          f"{hippocampus.n_sequences} sequences")

    # =====================================================================
    # Phase 2: Evaluate all conditions
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2: EVALUATE ALL CONDITIONS")
    print(f"{'=' * 60}")

    results = {}

    for domain in args.domains:
        print(f"\n--- {domain} ---")

        # Condition 1: Baseline (no LoRA, no injection)
        c1, t = evaluate_domain(base_model, tokenizer, domain)
        print(f"  Baseline:         {c1}/{t} ({c1/t:.1%})")

        # Condition 2: LoRA only (correct adapter)
        c2, t = evaluate_domain(trained_models[domain], tokenizer, domain)
        print(f"  LoRA only:        {c2}/{t} ({c2/t:.1%})")

        # Condition 3: Injection only (no LoRA)
        c3, t, avg_sim = evaluate_with_injection(
            base_model, tokenizer, domain, hippocampus, hook,
            injection_strength=args.injection_strength,
        )
        print(f"  Injection only:   {c3}/{t} ({c3/t:.1%}) "
              f"[avg recall sim={avg_sim:.3f}]")

        # Condition 4: LoRA + Injection
        c4, t, avg_sim4 = evaluate_with_injection(
            trained_models[domain], tokenizer, domain, hippocampus, hook,
            injection_strength=args.injection_strength,
        )
        print(f"  LoRA + Injection: {c4}/{t} ({c4/t:.1%}) "
              f"[avg recall sim={avg_sim4:.3f}]")

        # Condition 5: Wrong LoRA (control)
        wrong_domain = [d for d in args.domains if d != domain][0]
        c5, t = evaluate_domain(trained_models[wrong_domain], tokenizer, domain)
        print(f"  Wrong LoRA:       {c5}/{t} ({c5/t:.1%}) "
              f"[using {wrong_domain} adapter]")

        results[domain] = {
            "baseline": c1 / t,
            "lora_only": c2 / t,
            "injection_only": c3 / t,
            "lora_injection": c4 / t,
            "wrong_lora": c5 / t,
            "injection_recall_sim": avg_sim,
        }

    # =====================================================================
    # Cross-domain injection test
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 3: CROSS-DOMAIN INJECTION (emergent connections)")
    print(f"{'=' * 60}")
    print("Testing: does injection from one domain's training help another?")

    # For each domain, test injection on all OTHER domains
    print(f"\n  {'':>12s}", end="")
    for d in args.domains:
        print(f" {d:>10s}", end="")
    print("  (injection accuracy on each domain)")

    for eval_domain in args.domains:
        print(f"  {eval_domain:>12s}", end="")
        for inject_context in args.domains:
            # The injection comes from whatever the hippocampus recalls
            # given prompts from inject_context's domain
            c, t, _ = evaluate_with_injection(
                base_model, tokenizer, eval_domain, hippocampus, hook,
                injection_strength=args.injection_strength,
            )
            marker = " *" if inject_context == eval_domain else ""
            print(f" {c/t:>9.1%}{marker}", end="")
        print()

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  {'Domain':<12s} | {'Base':>6s} {'LoRA':>6s} {'Inject':>6s} "
          f"{'L+I':>6s} {'Wrong':>6s}")
    print(f"  {'-'*12}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}")
    for domain in args.domains:
        r = results[domain]
        print(f"  {domain:<12s} | {r['baseline']:>5.1%} {r['lora_only']:>5.1%} "
              f"{r['injection_only']:>5.1%} {r['lora_injection']:>5.1%} "
              f"{r['wrong_lora']:>5.1%}")

    # Check for emergent benefits
    any_injection_helped = any(
        results[d]["injection_only"] > results[d]["baseline"]
        for d in args.domains
    )
    any_synergy = any(
        results[d]["lora_injection"] > results[d]["lora_only"]
        for d in args.domains
    )

    if any_injection_helped:
        print("\n  >>> Hippocampal injection improved over baseline "
              "for at least one domain!")
    if any_synergy:
        print("  >>> LoRA + injection outperformed LoRA alone "
              "for at least one domain!")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--domains", nargs="+",
                        default=["capitals", "elements", "languages",
                                 "opposites", "colors"])
    parser.add_argument("--target_layer", type=int, default=-2,
                        help="Which layer to tap (negative = from end)")
    parser.add_argument("--injection_strength", type=float, default=0.5)
    parser.add_argument("--d_ec", type=int, default=384,
                        help="EC dimensionality")
    parser.add_argument("--k_ca3", type=int, default=50,
                        help="CA3 sparsity (k-winners)")
    parser.add_argument("--ca3_lr", type=float, default=1.0)
    parser.add_argument("--ca1_lr", type=float, default=50.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output_dir", default="hippocampal_checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
