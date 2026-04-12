"""
Quick test: run a single SVD REPL episode and see what the model does.
Also runs a baseline solve (no REPL, no SVD) for comparison.

    python test_single_episode.py
    python test_single_episode.py --problem_idx 100
    python test_single_episode.py --question "What is 15% of 240?"
    python test_single_episode.py --hard   # picks a harder problem
"""

import argparse
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from svd_repl import SVDManager, REPLEnvironment, extract_gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--question", default=None,
                        help="Custom question (skips GSM8K)")
    parser.add_argument("--problem_idx", type=int, default=None,
                        help="Specific GSM8K problem index")
    parser.add_argument("--hard", action="store_true",
                        help="Try to pick a harder problem (longer gold solution)")
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_directions", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking=true", "enable_thinking=false"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    print("Building SVD manager...")
    svd_manager = SVDManager(model, max_directions=args.max_directions)

    repl_env = REPLEnvironment(
        model=model,
        tokenizer=tokenizer,
        svd_manager=svd_manager,
        max_turns=args.max_turns,
    )

    # Get problem
    if args.question:
        question = args.question
        gold = "N/A"
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        if args.problem_idx is not None:
            idx = args.problem_idx
        elif args.hard:
            # Pick a problem with a longer solution (proxy for difficulty)
            candidates = [(i, len(row["answer"])) for i, row in enumerate(dataset)]
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Pick randomly from top 20% hardest
            top_hard = candidates[:len(candidates) // 5]
            idx = random.choice(top_hard)[0]
        else:
            idx = 42
        prob = dataset[idx]
        question = prob["question"]
        gold = prob["answer"]
        gold_num = extract_gold(gold)
        print(f"\nProblem index: {idx}")
        print(f"Gold answer: {gold_num}")

    print(f"\nQuestion: {question}")

    # --- BASELINE (no REPL, no SVD) ---
    print("\n" + "=" * 60)
    print("BASELINE SOLVE (no REPL, no SVD modifications)")
    print("=" * 60)
    baseline_text, baseline_pred, baseline_correct = repl_env.baseline_solve(
        question, gold
    )
    print(f"Baseline output (last 300 chars):\n{baseline_text[-300:]}")
    print(f"\nBaseline answer: {baseline_pred}")
    print(f"Baseline correct: {baseline_correct}")

    # --- REPL EPISODE ---
    print("\n" + "=" * 60)
    print("REPL EPISODE (with SVD tools)")
    print("=" * 60)
    episode = repl_env.run_episode(question, gold, run_baseline=False)
    # We already ran baseline above, store it
    episode.baseline_correct = baseline_correct
    episode.baseline_answer = baseline_pred

    # Print trajectory
    print("\n" + "-" * 60)
    print("FULL TRAJECTORY")
    print("-" * 60)
    for i, turn in enumerate(episode.turns):
        print(f"\n--- Turn {i+1} ({turn.role}) ---")
        if turn.code:
            print(f"[CODE]\n{turn.code}\n[/CODE]")
        content = turn.content
        if len(content) > 1200:
            content = content[:600] + "\n... (truncated) ...\n" + content[-300:]
        print(content)

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Question:            {question[:80]}...")
    print(f"Gold answer:         {extract_gold(gold)}")
    print(f"Baseline answer:     {baseline_pred} ({'CORRECT' if baseline_correct else 'WRONG'})")
    print(f"REPL answer:         {episode.final_answer} ({'CORRECT' if episode.correct else 'WRONG'})")
    print(f"Used tools:          {episode.used_tools}")
    print(f"Num SVD mods:        {episode.num_modifications}")
    print(f"Num turns:           {len(episode.turns)}")

    if episode.correct and not baseline_correct:
        print("\n*** SVD REPL HELPED: solved a problem baseline could not! ***")
    elif not episode.correct and baseline_correct:
        print("\n--- SVD REPL HURT: baseline solved it but REPL did not ---")
    elif episode.correct and baseline_correct:
        print("\n(both solved it)")
    else:
        print("\n(neither solved it)")


if __name__ == "__main__":
    main()