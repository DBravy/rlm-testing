"""
Quick test: run a single SVD REPL episode to see what the model does.

This lets you watch the model interact with its own weights before
committing to a full training run. Run with:

    python test_single_episode.py
    python test_single_episode.py --question "What is 15% of 240?"
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from svd_repl import SVDManager, REPLEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--question", default=None,
                        help="Custom question. If not provided, uses a GSM8K problem.")
    parser.add_argument("--max_turns", type=int, default=6)
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
    svd_manager = SVDManager(model, max_directions=10)

    repl_env = REPLEnvironment(
        model=model,
        tokenizer=tokenizer,
        svd_manager=svd_manager,
        max_turns=args.max_turns,
    )

    # Get a problem
    if args.question:
        question = args.question
        gold = "N/A"
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        prob = dataset[42]  # arbitrary fixed problem for reproducibility
        question = prob["question"]
        gold = prob["answer"]
        print(f"\nGold answer: {gold.split('####')[-1].strip()}")

    print(f"\nQuestion: {question}")
    print("=" * 60)
    print("RUNNING EPISODE")
    print("=" * 60)

    episode = repl_env.run_episode(question, gold)

    # Print the full trajectory
    print("\n" + "=" * 60)
    print("FULL TRAJECTORY")
    print("=" * 60)
    for i, turn in enumerate(episode.turns):
        print(f"\n--- Turn {i+1} ({turn.role}) ---")
        if turn.code:
            print(f"[CODE]\n{turn.code}")
            print(f"[END CODE]")
        print(turn.content[:1000])
        if len(turn.content) > 1000:
            print("... (truncated)")

    print(f"\n{'=' * 60}")
    print(f"Final answer: {episode.final_answer}")
    print(f"Gold answer:  {gold}")
    print(f"Correct:      {episode.correct}")
    print(f"Modifications applied: {episode.num_modifications}")
    print(f"Total turns:  {len(episode.turns)}")


if __name__ == "__main__":
    main()
