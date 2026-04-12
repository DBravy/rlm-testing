"""
SVD Self-Configuration Training on ARC-Easy (STaR-style)

1. Run episodes: model configures weights via REPL, answers science questions
2. Collect successful trajectories (where tools were used AND answer is correct)
3. Fine-tune on those trajectories
4. Repeat

Usage:
    python train_svd_agent.py
    python train_svd_agent.py --num_rounds 5 --episodes_per_round 50
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from svd_repl import SVDManager, REPLEnvironment, episode_to_messages


def collect_episodes(repl_env, problems, num_episodes, verbose=True):
    successes = []
    all_episodes = []
    sampled = [random.choice(problems) for _ in range(num_episodes)]

    for i, prob in enumerate(sampled):
        t0 = time.time()
        episode = repl_env.run_episode(
            prob["question"], prob["choices"], prob["answerKey"],
            run_baseline=True,
        )
        elapsed = time.time() - t0
        all_episodes.append(episode)

        # Only count as success if tools were actually used
        if episode.correct and episode.used_tools and episode.called_solve:
            successes.append(episode)

        if verbose:
            status = "CORRECT" if episode.correct else "wrong"
            base = "base:Y" if episode.baseline_correct else "base:N"
            tools = "svd:Y" if episode.used_tools else "svd:N"
            solved = "solve:Y" if episode.called_solve else "solve:N"
            print(f"  [{i+1}/{num_episodes}] {status} | {base} | {tools} | "
                  f"{solved} | {episode.num_modifications} mods | "
                  f"{elapsed:.1f}s | pred={episode.final_answer} "
                  f"gold={episode.gold_label}")

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
        print("  No qualifying episodes for SFT, skipping.")
        return

    print(f"  SFT on {len(episodes)} episodes...")
    samples = prepare_sft_dataset(episodes, tokenizer)
    dataset = Dataset.from_dict({
        "input_ids": [s["input_ids"] for s in samples],
        "labels": [s["labels"] for s in samples],
        "attention_mask": [s["attention_mask"] for s in samples],
    })
    dataset.set_format("torch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
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
        ),
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

    # Load ARC-Easy
    arc_name = "ARC-Easy" if args.split == "easy" else "ARC-Challenge"
    print(f"Loading {arc_name}...")
    dataset = load_dataset("allenai/ai2_arc", arc_name, split="train")
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

        n = len(all_episodes)
        n_correct = sum(1 for e in all_episodes if e.correct)
        n_baseline = sum(1 for e in all_episodes if e.baseline_correct)
        n_tools = sum(1 for e in all_episodes if e.used_tools)
        n_solved = sum(1 for e in all_episodes if e.called_solve)
        n_qualified = len(successes)  # correct AND used tools AND called solve
        n_helped = sum(1 for e in all_episodes
                       if e.correct and not e.baseline_correct
                       and e.used_tools)
        n_hurt = sum(1 for e in all_episodes
                     if not e.correct and e.baseline_correct)

        print(f"\nRound {round_idx+1} summary:")
        print(f"  Total correct:        {n_correct}/{n} ({n_correct/n:.1%})")
        print(f"  Baseline correct:     {n_baseline}/{n} ({n_baseline/n:.1%})")
        print(f"  Used scale_direction: {n_tools}/{n}")
        print(f"  Called solve():       {n_solved}/{n}")
        print(f"  Qualified for SFT:    {n_qualified}/{n} (correct + tools + solve)")
        print(f"  SVD helped:           {n_helped}")
        print(f"  SVD hurt:             {n_hurt}")

        cumulative_successes.extend(successes)

        stats = {
            "round": round_idx + 1, "total": n,
            "correct": n_correct, "baseline_correct": n_baseline,
            "used_tools": n_tools, "called_solve": n_solved,
            "qualified_sft": n_qualified,
            "helped": n_helped, "hurt": n_hurt,
            "cumulative_sft": len(cumulative_successes),
        }
        all_stats.append(stats)

        # SFT
        if successes:
            train_episodes = successes
            if len(cumulative_successes) > len(successes):
                older = [e for e in cumulative_successes if e not in successes]
                k = min(len(older), len(successes))
                if k > 0:
                    train_episodes = successes + random.sample(older, k)
            sft_step(model, tokenizer, train_episodes, str(output_dir),
                     round_idx, lr=args.learning_rate)

        save_path = output_dir / f"round_{round_idx}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for s in all_stats:
        print(f"  Round {s['round']}: "
              f"correct={s['correct']}/{s['total']} "
              f"baseline={s['baseline_correct']}/{s['total']} "
              f"tools={s['used_tools']} solve={s['called_solve']} "
              f"sft_qualified={s['qualified_sft']} "
              f"helped={s['helped']} hurt={s['hurt']}")

    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {output_dir / 'training_stats.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--split", default="easy", choices=["easy", "challenge"])
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--episodes_per_round", type=int, default=50)
    parser.add_argument("--max_problems", type=int, default=500)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_directions", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="svd_agent_checkpoints")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
