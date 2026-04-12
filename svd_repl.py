"""
SVD REPL Environment for Self-Configuring Language Models
"""

import re
import io
import sys
import traceback
from dataclasses import dataclass, field

import torch

# Code block delimiter, built this way so it doesn't interfere with any
# outer templating or file-creation tools.
CB = chr(96) * 3  # triple backtick


# ---------------------------------------------------------------------------
# SVD Manager
# ---------------------------------------------------------------------------

class SVDManager:
    DEFAULT_MATRICES = [
        "self_attn.o_proj", "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate_proj",
    ]

    def __init__(self, model, matrix_names=None, max_directions=20):
        self.model = model
        self.matrix_names = matrix_names or self.DEFAULT_MATRICES
        self.max_directions = max_directions
        self.num_layers = len(model.model.layers)
        self.svd_cache = {}
        self.active_deltas = []
        self._precompute_svds()

    def _get_weight(self, layer_idx, matrix_name):
        layer = self.model.model.layers[layer_idx]
        parts = matrix_name.split(".")
        obj = layer
        for part in parts:
            obj = getattr(obj, part)
        return obj.weight

    def _precompute_svds(self):
        print(f"Precomputing SVDs for {self.num_layers} layers, "
              f"{len(self.matrix_names)} matrices each...")
        for layer_idx in range(self.num_layers):
            for matrix_name in self.matrix_names:
                try:
                    weight = self._get_weight(layer_idx, matrix_name)
                    W = weight.detach().float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    k = min(self.max_directions, len(S))
                    self.svd_cache[(layer_idx, matrix_name)] = (
                        U[:, :k].cpu(), S[:k].cpu(), Vt[:k, :].cpu(),
                    )
                except Exception as e:
                    print(f"  Warning: layer {layer_idx} {matrix_name}: {e}")
        print(f"  Cached {len(self.svd_cache)} decompositions")

    def get_spectrum(self, layer_idx, matrix_name):
        key = (layer_idx, matrix_name)
        if key not in self.svd_cache:
            raise ValueError(f"No SVD cached for layer {layer_idx}, {matrix_name}")
        _, S, _ = self.svd_cache[key]
        return S.tolist()

    def scale_direction(self, layer_idx, matrix_name, direction_idx, scale_factor):
        key = (layer_idx, matrix_name)
        if key not in self.svd_cache:
            raise ValueError(f"No SVD cached for layer {layer_idx}, {matrix_name}")
        U, S, Vt = self.svd_cache[key]
        if direction_idx >= len(S):
            raise ValueError(f"Direction {direction_idx} out of range (max {len(S)-1})")
        scale_factor = max(-5.0, min(scale_factor, 20.0))
        weight = self._get_weight(layer_idx, matrix_name)
        sigma = S[direction_idx].to(weight.dtype).to(weight.device)
        u = U[:, direction_idx].to(weight.dtype).to(weight.device)
        v = Vt[direction_idx, :].to(weight.dtype).to(weight.device)
        delta = (scale_factor - 1.0) * sigma * torch.outer(u, v)
        weight.data.add_(delta)
        self.active_deltas.append((weight, delta))
        return f"Scaled layer {layer_idx} {matrix_name} direction {direction_idx} by {scale_factor}"

    def reset_all(self):
        count = len(self.active_deltas)
        for weight, delta in reversed(self.active_deltas):
            weight.data.sub_(delta)
        self.active_deltas.clear()
        return f"Reset {count} modifications"

    def list_layers(self):
        return {
            "num_layers": self.num_layers,
            "matrices": self.matrix_names,
            "max_directions": self.max_directions,
        }


# ---------------------------------------------------------------------------
# System prompt with few-shot examples
# ---------------------------------------------------------------------------

def build_system_prompt(max_turns):
    """
    Builds the system prompt with embedded few-shot examples.
    Constructed as a function so we can use CB for code fences cleanly.
    """
    return (
        "You are a self-configuring language model. You have a Python REPL "
        "that lets you inspect and modify your own weights via SVD direction "
        "scaling, then solve a math problem with those modified weights.\n"
        "\n"
        "IMPORTANT: You MUST use the REPL. Do NOT solve the problem in your "
        "head or output an answer directly. Your job is to:\n"
        "1. Explore the model structure\n"
        "2. Configure weights using scale_direction()\n"
        "3. Call solve(question) to get an answer with your modified weights\n"
        "4. Output FINAL(number) with the result\n"
        "\n"
        "Available functions:\n"
        "  list_layers()                                          -> model info\n"
        "  get_spectrum(layer_idx, matrix_name)                   -> singular values\n"
        "  scale_direction(layer_idx, matrix_name, dir_idx, factor) -> apply scaling\n"
        "  reset_all()                                            -> undo all mods\n"
        "  solve(question)                                        -> answer with current weights\n"
        "\n"
        f"Write code in {CB}python ... {CB} blocks. When done: FINAL(number)\n"
        f"Max {max_turns} code blocks.\n"
        "\n"
        "=== EXAMPLE INTERACTION ===\n"
        "\n"
        "User: Math problem to solve:\n"
        "\n"
        "A store sells 45 widgets at $3 each. What is the total revenue?\n"
        "\n"
        "Assistant: I will explore the model and configure weights for this "
        "arithmetic task.\n"
        "\n"
        f"{CB}python\n"
        "info = list_layers()\n"
        "print(f\"Layers: {info['num_layers']}\")\n"
        "print(f\"Matrices: {info['matrices'][:3]}\")\n"
        f"{CB}\n"
        "\n"
        "User: REPL output:\n"
        f"{CB}\n"
        "Layers: 28\n"
        "Matrices: ['self_attn.o_proj', 'self_attn.q_proj', 'self_attn.k_proj']\n"
        f"{CB}\n"
        "\n"
        "Assistant: Let me check singular values in a mid-layer and apply scaling.\n"
        "\n"
        f"{CB}python\n"
        "spec = get_spectrum(14, \"self_attn.o_proj\")\n"
        "print(f\"Top 5 values: {spec[:5]}\")\n"
        "print(scale_direction(14, \"self_attn.o_proj\", 0, 1.5))\n"
        "print(scale_direction(14, \"mlp.up_proj\", 1, 0.7))\n"
        f"{CB}\n"
        "\n"
        "User: REPL output:\n"
        f"{CB}\n"
        "Top 5 values: [142.3, 89.1, 45.7, 32.1, 28.4]\n"
        "Scaled layer 14 self_attn.o_proj direction 0 by 1.5\n"
        "Scaled layer 14 mlp.up_proj direction 1 by 0.7\n"
        f"{CB}\n"
        "\n"
        "Assistant: Good. Now solving with modified weights.\n"
        "\n"
        f"{CB}python\n"
        "answer = solve(\"A store sells 45 widgets at $3 each. "
        "What is the total revenue?\")\n"
        "print(answer)\n"
        f"{CB}\n"
        "\n"
        "User: REPL output:\n"
        f"{CB}\n"
        "45 widgets times $3 each equals $135. The total revenue is $135.\n"
        "#### 135\n"
        f"{CB}\n"
        "\n"
        "Assistant: FINAL(135)\n"
        "\n"
        "=== END EXAMPLE ===\n"
        "\n"
        "Now solve the given problem. Start by exploring, configure weights, "
        "call solve(), then output FINAL(number)."
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str           # "assistant" or "tool"
    content: str
    code: str = ""


@dataclass
class Episode:
    question: str
    gold_answer: str
    turns: list = field(default_factory=list)
    final_answer: str = None
    correct: bool = False
    num_modifications: int = 0
    used_tools: bool = False
    baseline_correct: bool = False
    baseline_answer: str = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_thinking(text):
    """Remove <think>...</think> blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def extract_number(text):
    """Extract the last number from text."""
    if text is None:
        return None
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_gold(gold_answer):
    """Extract gold number from GSM8K answer format."""
    match = re.search(r"####\s*(-?[\d,]+)", gold_answer)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", gold_answer)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(pred, gold):
    """Check if predicted and gold answers match."""
    if pred is None or gold is None:
        return False
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# REPL Environment
# ---------------------------------------------------------------------------

class REPLEnvironment:
    def __init__(self, model, tokenizer, svd_manager, max_turns=8,
                 max_new_tokens=1024, solve_max_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.svd_manager = svd_manager
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.solve_max_tokens = solve_max_tokens

    def _generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_thinking(text)

    def _solve_math(self, question):
        """Run model on a math problem with CURRENT (possibly modified) weights."""
        messages = [
            {"role": "user", "content": (
                "Solve this math problem. Show your work step by step. "
                "End with your final numerical answer after '####'.\n\n"
                + question
            )}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.solve_max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_thinking(text)

    def baseline_solve(self, question, gold_answer):
        """
        Solve with unmodified weights, no REPL. Returns (text, pred, correct).
        For tracking whether SVD/REPL actually helped.
        """
        self.svd_manager.reset_all()
        answer_text = self._solve_math(question)
        pred = extract_number(answer_text)
        gold = extract_gold(gold_answer)
        correct = answers_match(pred, gold)
        return answer_text, pred, correct

    def _extract_code(self, text):
        # Try python-tagged blocks first
        pattern = CB + r"python\s*\n(.*?)" + CB
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        # Try untagged blocks
        pattern = CB + r"\s*\n(.*?)" + CB
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return None

    def _extract_final(self, text):
        match = re.search(r"FINAL\((.+?)\)", text, re.DOTALL)
        if match:
            return match.group(1).strip().strip("'\"")
        return None

    def _execute_code(self, code, namespace):
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            exec(code, namespace)
            sys.stdout = old_stdout
            output = stdout_capture.getvalue()
            if not output.strip():
                output = "(no output)"
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            return output
        except Exception:
            sys.stdout = old_stdout
            return f"Error:\n{traceback.format_exc()}"

    def run_episode(self, question, gold_answer, run_baseline=True):
        episode = Episode(question=question, gold_answer=gold_answer)

        # Baseline comparison
        if run_baseline:
            _, baseline_pred, baseline_correct = self.baseline_solve(
                question, gold_answer
            )
            episode.baseline_correct = baseline_correct
            episode.baseline_answer = baseline_pred

        # REPL namespace
        namespace = {
            "list_layers": self.svd_manager.list_layers,
            "get_spectrum": self.svd_manager.get_spectrum,
            "scale_direction": self.svd_manager.scale_direction,
            "reset_all": self.svd_manager.reset_all,
            "solve": self._solve_math,
            "print": print,
        }

        system = build_system_prompt(self.max_turns)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Math problem to solve:\n\n{question}"},
        ]

        # REPL loop
        for turn_idx in range(self.max_turns):
            response = self._generate(messages)

            # Check for final answer
            final = self._extract_final(response)
            if final is not None:
                episode.turns.append(Turn(role="assistant", content=response))
                episode.final_answer = final
                break

            # Check for code
            code = self._extract_code(response)
            if code is None:
                episode.turns.append(Turn(role="assistant", content=response))
                messages.append({"role": "assistant", "content": response})
                nudge = (
                    "You must write Python code in a " + CB + "python block to "
                    "use the REPL tools. Do not solve the problem directly. "
                    "Use the tools to configure weights, then call solve(question)."
                )
                messages.append({"role": "user", "content": nudge})
                continue

            # Execute code
            episode.used_tools = True
            episode.turns.append(Turn(role="assistant", content=response, code=code))
            messages.append({"role": "assistant", "content": response})

            output = self._execute_code(code, namespace)
            episode.turns.append(Turn(role="tool", content=output))
            messages.append({"role": "user",
                             "content": f"REPL output:\n{CB}\n{output}\n{CB}"})

        # Fallback: extract from last turn if no FINAL
        if episode.final_answer is None and episode.turns:
            episode.final_answer = extract_number(episode.turns[-1].content)

        # Score
        episode.num_modifications = len(self.svd_manager.active_deltas)
        pred = extract_number(episode.final_answer)
        gold = extract_gold(gold_answer)
        episode.correct = answers_match(pred, gold)

        # Reset
        self.svd_manager.reset_all()
        return episode


# ---------------------------------------------------------------------------
# Trajectory formatting for SFT
# ---------------------------------------------------------------------------

def episode_to_messages(episode):
    system = build_system_prompt(8)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Math problem to solve:\n\n{episode.question}"},
    ]
    for turn in episode.turns:
        if turn.role == "assistant":
            messages.append({"role": "assistant", "content": turn.content})
        elif turn.role == "tool":
            messages.append({"role": "user",
                             "content": f"REPL output:\n{CB}\n{turn.content}\n{CB}"})
    return messages