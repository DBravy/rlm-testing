"""
Quick test: run a single SVD REPL episode on ARC-Easy.

    python test_single_episode.py
    python test_single_episode.py --problem_idx 100
    python test_single_episode.py --split challenge  # harder questions
"""

import argparse
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from svd_repl import SVDManager, REPLEnvironment, format_arc_question, normalize_gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--problem_idx", type=int, default=None)
    parser.add_argument("--split", default="easy",
                        choices=["easy", "challenge"],
                        help="ARC-Easy or ARC-Challenge")
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
        model=model, tokenizer=tokenizer, svd_manager=svd_manager,
        max_turns=args.max_turns,
    )

    # Load ARC
    arc_name = "ARC-Easy" if args.split == "easy" else "ARC-Challenge"
    print(f"Loading {arc_name}...")
    dataset = load_dataset("allenai/ai2_arc", arc_name, split="train")

    idx = args.problem_idx if args.problem_idx is not None else random.randint(0, len(dataset)-1)
    prob = dataset[idx]

    question = prob["question"]
    choices = prob["choices"]
    gold_label = normalize_gold(choices, prob["answerKey"])

    q_text = format_arc_question(question, choices)
    raw_idx = choices["label"].index(prob["answerKey"])
    gold_text = choices["text"][raw_idx]

    print(f"\nProblem index: {idx}")
    print(f"Gold answer: {gold_label}) {gold_text}")
    print(f"\n{q_text}")

    # --- BASELINE ---
    print("\n" + "=" * 60)
    print("BASELINE (no REPL, no SVD)")
    print("=" * 60)
    baseline_text, baseline_pred, baseline_correct = repl_env.baseline_solve(
        question, choices, gold_label
    )
    print(f"Model output (last 300 chars):\n{baseline_text[-300:]}")
    print(f"\nBaseline answer: {baseline_pred} ({'CORRECT' if baseline_correct else 'WRONG'})")

    # --- REPL EPISODE ---
    print("\n" + "=" * 60)
    print("REPL EPISODE (with SVD tools)")
    print("=" * 60)
    episode = repl_env.run_episode(question, choices, gold_label, run_baseline=False)
    episode.baseline_correct = baseline_correct
    episode.baseline_answer = baseline_pred

    print("\n" + "-" * 60)
    print("TRAJECTORY")
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
    print(f"Question:        {question[:80]}...")
    print(f"Gold:            {gold_label}) {gold_text}")
    print(f"Baseline:        {baseline_pred} ({'CORRECT' if baseline_correct else 'WRONG'})")
    print(f"REPL answer:     {episode.final_answer} ({'CORRECT' if episode.correct else 'WRONG'})")
    print(f"Used SVD tools:  {episode.used_tools}")
    print(f"Called solve():  {episode.called_solve}")
    print(f"Num SVD mods:    {episode.num_modifications}")
    print(f"Num turns:       {len(episode.turns)}")

    if episode.correct and not baseline_correct:
        print("\n*** SVD REPL HELPED ***")
    elif not episode.correct and baseline_correct:
        print("\n--- SVD REPL HURT ---")
    elif episode.correct and baseline_correct:
        print("\n(both correct)")
    else:
        print("\n(neither correct)")


if __name__ == "__main__":
    main()
