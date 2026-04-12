"""
SVD REPL Environment for Self-Configuring Language Models

Provides:
  - SVDManager: precomputes SVDs, applies/undoes rank-1 weight modifications
  - REPLEnvironment: multi-turn code execution loop with tool functions
  - EpisodeRunner: runs a full configure-then-solve episode and returns trajectory

The model is given a math problem and access to tool functions that let it
inspect and modify its own weights via SVD direction scaling. It writes Python
code to configure itself, then the modified model solves the problem.
"""

import re
import io
import sys
import copy
import traceback
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# SVD Manager
# ---------------------------------------------------------------------------

class SVDManager:
    """
    Precomputes SVDs for selected weight matrices and provides fast
    rank-1 modifications via singular direction scaling.
    """

    # Which matrices to decompose in each transformer layer
    DEFAULT_MATRICES = [
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
    ]

    def __init__(self, model, matrix_names=None, max_directions=20):
        self.model = model
        self.matrix_names = matrix_names or self.DEFAULT_MATRICES
        self.max_directions = max_directions
        self.num_layers = len(model.model.layers)

        # Cache: (layer_idx, matrix_name) -> (U, S, Vt)
        self.svd_cache = {}
        # Track active modifications for reset
        self.active_deltas = []  # list of (weight_tensor, delta_tensor)

        self._precompute_svds()

    def _get_weight(self, layer_idx, matrix_name):
        """Get a reference to a weight tensor."""
        layer = self.model.model.layers[layer_idx]
        parts = matrix_name.split(".")
        obj = layer
        for part in parts:
            obj = getattr(obj, part)
        return obj.weight

    def _precompute_svds(self):
        """Compute and cache SVDs for all target matrices."""
        print(f"Precomputing SVDs for {self.num_layers} layers, "
              f"{len(self.matrix_names)} matrices each...")
        for layer_idx in range(self.num_layers):
            for matrix_name in self.matrix_names:
                try:
                    weight = self._get_weight(layer_idx, matrix_name)
                    W = weight.detach().float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    # Only keep top-k directions to save memory
                    k = min(self.max_directions, len(S))
                    self.svd_cache[(layer_idx, matrix_name)] = (
                        U[:, :k].cpu(),
                        S[:k].cpu(),
                        Vt[:k, :].cpu(),
                    )
                except Exception as e:
                    print(f"  Warning: could not decompose layer {layer_idx} "
                          f"{matrix_name}: {e}")
        print(f"  Cached {len(self.svd_cache)} decompositions")

    def get_spectrum(self, layer_idx, matrix_name):
        """Return the top singular values for a given weight matrix."""
        key = (layer_idx, matrix_name)
        if key not in self.svd_cache:
            raise ValueError(f"No SVD cached for layer {layer_idx}, {matrix_name}")
        _, S, _ = self.svd_cache[key]
        return S.tolist()

    def scale_direction(self, layer_idx, matrix_name, direction_idx, scale_factor):
        """
        Apply a rank-1 modification: scale a singular direction by the given factor.
        W' = W + (scale - 1) * sigma_i * u_i @ v_i^T
        """
        key = (layer_idx, matrix_name)
        if key not in self.svd_cache:
            raise ValueError(f"No SVD cached for layer {layer_idx}, {matrix_name}")

        U, S, Vt = self.svd_cache[key]
        if direction_idx >= len(S):
            raise ValueError(f"Direction {direction_idx} out of range (max {len(S)-1})")

        # Clamp scale factor to prevent catastrophic modifications
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
        """Undo all active modifications."""
        count = len(self.active_deltas)
        for weight, delta in reversed(self.active_deltas):
            weight.data.sub_(delta)
        self.active_deltas.clear()
        return f"Reset {count} modifications"

    def list_layers(self):
        """Return available layers and matrices."""
        return {
            "num_layers": self.num_layers,
            "matrices": self.matrix_names,
            "max_directions": self.max_directions,
        }


# ---------------------------------------------------------------------------
# REPL Environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You have access to a Python REPL with tools for inspecting and modifying your own neural network weights via SVD (Singular Value Decomposition) direction scaling.

Available functions in the REPL:
- list_layers() -> dict with num_layers, matrix names, max_directions
- get_spectrum(layer_idx: int, matrix_name: str) -> list of top singular values
- scale_direction(layer_idx: int, matrix_name: str, direction_idx: int, scale_factor: float) -> confirmation string
- reset_all() -> undoes all weight modifications
- solve(question: str) -> runs the math problem with your CURRENT weights and returns the model's answer

Your task:
1. You will be given a math problem.
2. Use the REPL tools to configure your weights for the task.
3. Call solve(question) to generate your answer with the modified weights.
4. When you have your final answer, output: FINAL(your_answer)

Write Python code in ```python ... ``` blocks. You will see the output after each block.
Keep your configuration focused. You have a maximum of {max_turns} code executions.
Start by exploring what's available, then make targeted modifications."""


@dataclass
class Turn:
    """One turn of the REPL interaction."""
    role: str           # "assistant" or "tool"
    content: str
    code: str = ""      # if role is "assistant", the extracted code


@dataclass
class Episode:
    """A complete configure-then-solve episode."""
    question: str
    gold_answer: str
    turns: list = field(default_factory=list)
    final_answer: str = None
    correct: bool = False
    num_modifications: int = 0


class REPLEnvironment:
    """
    Multi-turn REPL loop. The model generates text with code blocks,
    we execute the code and feed stdout back, until the model outputs
    FINAL(...) or we hit the turn limit.
    """

    def __init__(self, model, tokenizer, svd_manager, max_turns=8,
                 max_new_tokens=1024, solve_max_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.svd_manager = svd_manager
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.solve_max_tokens = solve_max_tokens

    def _generate(self, messages):
        """Generate a response from the model given a message history."""
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
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _solve_math(self, question):
        """
        Run the model on a math problem with its CURRENT weights
        (which may be SVD-modified). This is the 'inner' evaluation.
        """
        messages = [
            {"role": "user", "content": f"Solve this math problem step by step. "
             f"End with your final numerical answer on its own line.\n\n{question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.solve_max_tokens,
                do_sample=False,  # greedy for evaluation
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _extract_code(self, text):
        """Extract Python code from markdown code blocks."""
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        # Also try generic code blocks
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return None

    def _extract_final(self, text):
        """Extract the final answer from FINAL(...) tags."""
        match = re.search(r"FINAL\((.+?)\)", text, re.DOTALL)
        if match:
            return match.group(1).strip().strip("'\"")
        return None

    def _execute_code(self, code, namespace):
        """Execute code in a sandboxed namespace, capturing stdout."""
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            exec(code, namespace)
            sys.stdout = old_stdout
            output = stdout_capture.getvalue()
            if not output.strip():
                output = "(no output)"
            # Truncate very long output
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            return output
        except Exception:
            sys.stdout = old_stdout
            return f"Error:\n{traceback.format_exc()}"

    def run_episode(self, question, gold_answer):
        """
        Run a complete episode: the model configures itself via REPL,
        then solves the math problem.
        """
        episode = Episode(question=question, gold_answer=gold_answer)

        # Build the REPL namespace with tool functions
        namespace = {
            "list_layers": self.svd_manager.list_layers,
            "get_spectrum": self.svd_manager.get_spectrum,
            "scale_direction": self.svd_manager.scale_direction,
            "reset_all": self.svd_manager.reset_all,
            "solve": self._solve_math,
            "print": print,  # ensure print is available
        }

        # System prompt + user message
        system = SYSTEM_PROMPT.format(max_turns=self.max_turns)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Math problem to solve:\n\n{question}"},
        ]

        # REPL interaction loop
        for turn_idx in range(self.max_turns):
            # Generate model response
            response = self._generate(messages)

            # Check for final answer
            final = self._extract_final(response)
            if final is not None:
                episode.turns.append(Turn(role="assistant", content=response))
                episode.final_answer = final
                break

            # Check for code to execute
            code = self._extract_code(response)
            if code is None:
                # Model didn't write code or give a final answer,
                # nudge it to do one or the other
                episode.turns.append(Turn(role="assistant", content=response))
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content":
                    "Please either write Python code in a ```python block "
                    "or provide your final answer as FINAL(answer)."})
                continue

            # Execute the code
            episode.turns.append(Turn(role="assistant", content=response, code=code))
            messages.append({"role": "assistant", "content": response})

            output = self._execute_code(code, namespace)
            episode.turns.append(Turn(role="tool", content=output))
            messages.append({"role": "user", "content": f"REPL output:\n```\n{output}\n```"})

        # If we exhausted turns without a FINAL, try to extract answer from last response
        if episode.final_answer is None and episode.turns:
            last_text = episode.turns[-1].content
            numbers = re.findall(r"-?\d+\.?\d*", last_text)
            if numbers:
                episode.final_answer = numbers[-1]

        # Check correctness
        episode.num_modifications = len(self.svd_manager.active_deltas)
        episode.correct = self._check_answer(episode.final_answer, gold_answer)

        # Always reset weights after an episode
        self.svd_manager.reset_all()

        return episode

    def _check_answer(self, predicted, gold):
        """Check if the predicted answer matches the gold answer."""
        if predicted is None or gold is None:
            return False
        # Extract number from gold (GSM8K format)
        gold_match = re.search(r"####\s*(-?[\d,]+)", gold)
        if gold_match:
            gold_num = gold_match.group(1).replace(",", "")
        else:
            gold_nums = re.findall(r"-?\d+\.?\d*", gold)
            gold_num = gold_nums[-1] if gold_nums else None

        pred_nums = re.findall(r"-?\d+\.?\d*", predicted)
        pred_num = pred_nums[-1] if pred_nums else None

        if gold_num is None or pred_num is None:
            return False

        # Try exact match first
        if pred_num == gold_num:
            return True
        # Try float comparison for approximate matches
        try:
            return abs(float(pred_num) - float(gold_num)) < 1e-6
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Trajectory formatting (for SFT on successful episodes)
# ---------------------------------------------------------------------------

def episode_to_messages(episode):
    """
    Convert a successful episode into a list of chat messages
    suitable for SFT training.
    """
    system = SYSTEM_PROMPT.format(max_turns=8)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Math problem to solve:\n\n{episode.question}"},
    ]
    for turn in episode.turns:
        if turn.role == "assistant":
            messages.append({"role": "assistant", "content": turn.content})
        elif turn.role == "tool":
            messages.append({"role": "user", "content": f"REPL output:\n```\n{turn.content}\n```"})
    return messages
