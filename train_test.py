import re
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# Load GSM8K and format it for GRPO
dataset = load_dataset("openai/gsm8k", "main", split="train")

def format_prompt(example):
    return {
        "prompt": [
            {"role": "user", "content": example["question"]}
        ],
        "answer": example["answer"]
    }

dataset = dataset.map(format_prompt)

# Extract the final number from a GSM8K answer string
def extract_answer(text):
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    # Also try to grab the last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None

# Reward function: 1.0 if correct, 0.0 if wrong
def math_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer):
        # completions are chat message lists, e.g. [{"role": "assistant", "content": "..."}]
        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion
        gold = extract_answer(ans)
        pred = extract_answer(text)
        rewards.append(1.0 if pred and gold and pred == gold else 0.0)
    return rewards

# Training config
config = GRPOConfig(
    output_dir="qwen3-06b-math-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,          # completions per prompt
    max_completion_length=256,  # max tokens per completion
    logging_steps=1,
    save_steps=50,
    learning_rate=1e-5,
    bf16=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=math_reward,
    args=config,
    train_dataset=dataset,
)

trainer.train()