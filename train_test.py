import re
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

MODEL = "Qwen/Qwen3-0.6B"

# Load tokenizer and disable thinking mode
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "enable_thinking=true", "enable_thinking=false"
    )

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
    max_steps=200,                  # cap at ~2 hours instead of full epoch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=8,              # more completions = better advantage estimates
    max_completion_length=1024,     # room for chain-of-thought + final answer
    max_prompt_length=256,
    logging_steps=1,
    save_steps=50,
    learning_rate=5e-6,             # slightly lower, less aggressive for small model
    bf16=True,
    log_completions=True,           # print actual completions so you can see what it's doing
)

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=math_reward,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,     # pass our modified tokenizer
)

trainer.train()