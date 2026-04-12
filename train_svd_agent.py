"""
SVD Self-Configuration Training (STaR-style)

1. Run episodes where the model configures its weights via REPL, then solves math
2. Collect successful trajectories
3. Fine-tune on those trajectories
4. Repeat

Also tracks baseline performance (no REPL) for comparison at each round.

Usage:
    python train_svd_agent.py
    python train_svd_agent.py --num_rounds 5 --episodes_per_round 50
    python train_svd_agent.py --no_lora  # full fine-tuning
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from svd_repl import SVDManager, REPLEnvironment, episode_to_messages


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def collect_episodes(repl_env, problems, num_episodes, verbose=True):
    successes = []
    all_episodes = []
    sampled = [random.choice(problems) for _ in range(num_episodes)]

    for i, prob in enumerate(sampled):
        t0 = time.time()
        episode = repl_env.run_episode(
            prob["question"], prob["answer"], run_baseline=True
        )
        elapsed = time.time() - t0
        all_episodes.append(episode)
        if episode.correct:
            successes.append(episode)

        if verbose:
            status = "CORRECT" if episode.correct else "wrong"
            base = "base:Y" if episode.baseline_correct else "base:N"
            tools = "tools:Y" if episode.used_tools else "tools:N"
            mods = episode.num_modifications
            print(f"  [{i+1}/{num_episodes}] {status} | {base} | {tools} | "
                  f"{mods} mods | {elapsed:.1f}s | "
                  f"pred={episode.final_answer}")

    return successes, all_episodes


def prepare_sft_dataset(episodes, tokenizer, max_length=2048):
    samples = []
    for ep in episodes:
        messages = episode_to_messages(ep)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        tokens = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        samples.append({
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        })
    return samples


def sft_step(model, tokenizer, episodes, output_dir, round_idx, lr=2e-5):
    if not episodes:
        print("  No successful episodes to train on, skipping SFT.")
        return

    print(f"  Preparing SFT from {len(episodes)} successful episodes...")
    samples = prepare_sft_dataset(episodes, tokenizer)

    dataset = Dataset.from_dict({
        "input_ids": [s["input_ids"] for s in samples],
        "labels": [s["labels"] for s in samples],
        "attention_mask": [s["attention_mask"] for s in samples],
    })
    dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/round_{round_idx}",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt",
        ),
    )
    trainer.train()
    print(f"  SFT complete for round {round_idx}")


def main(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking=true", "enable_thinking=false"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    if args.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj",
                            "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("Building SVD manager...")
    base_model = model.base_model.model if args.use_lora else model
    svd_manager = SVDManager(base_model, max_directions=args.max_directions)

    repl_env = REPLEnvironment(
        model=model, tokenizer=tokenizer, svd_manager=svd_manager,
        max_turns=args.max_turns,
    )

    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    problems = list(dataset.select(range(min(args.max_problems, len(dataset)))))
    random.shuffle(problems)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    cumulative_successes = []

    for round_idx in range(args.num_rounds):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_idx + 1}/{args.num_rounds}")
        print(f"{'=' * 60}")

        successes, all_episodes = collect_episodes(
            repl_env, problems, args.episodes_per_round
        )

        # Compute stats
        n = len(all_episodes)
        n_correct = len(successes)
        n_baseline_correct = sum(1 for e in all_episodes if e.baseline_correct)
        n_used_tools = sum(1 for e in all_episodes if e.used_tools)
        n_helped = sum(1 for e in all_episodes
                       if e.correct and not e.baseline_correct)
        n_hurt = sum(1 for e in all_episodes
                     if not e.correct and e.baseline_correct)

        print(f"\nRound {round_idx+1} summary:")
        print(f"  REPL correct:     {n_correct}/{n} ({n_correct/n:.1%})")
        print(f"  Baseline correct: {n_baseline_correct}/{n} ({n_baseline_correct/n:.1%})")
        print(f"  Used tools:       {n_used_tools}/{n} ({n_used_tools/n:.1%})")
        print(f"  REPL helped:      {n_helped} (solved what baseline couldn't)")
        print(f"  REPL hurt:        {n_hurt} (baseline solved, REPL didn't)")

        cumulative_successes.extend(successes)

        stats = {
            "round": round_idx + 1,
            "total": n,
            "repl_correct": n_correct,
            "baseline_correct": n_baseline_correct,
            "used_tools": n_used_tools,
            "repl_helped": n_helped,
            "repl_hurt": n_hurt,
            "cumulative_successes": len(cumulative_successes),
        }
        all_stats.append(stats)

        # SFT on successes
        if successes:
            train_episodes = successes
            if len(cumulative_successes) > len(successes):
                older = [e for e in cumulative_successes if e not in successes]
                sample_size = min(len(older), len(successes))
                if sample_size > 0:
                    train_episodes = successes + random.sample(older, sample_size)

            print(f"\nSFT on {len(train_episodes)} episodes...")
            sft_step(model, tokenizer, train_episodes, str(output_dir),
                     round_idx, lr=args.learning_rate)

        # Save
        save_path = output_dir / f"round_{round_idx}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

    # Final summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for s in all_stats:
        print(f"  Round {s['round']}: "
              f"repl={s['repl_correct']}/{s['total']} "
              f"baseline={s['baseline_correct']}/{s['total']} "
              f"tools_used={s['used_tools']}/{s['total']} "
              f"helped={s['repl_helped']} hurt={s['repl_hurt']}")

    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--episodes_per_round", type=int, default=50)
    parser.add_argument("--max_problems", type=int, default=200)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_directions", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="svd_agent_checkpoints")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())