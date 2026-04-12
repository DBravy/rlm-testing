"""
SVD REPL Environment for Self-Configuring Language Models
Adapted for ARC (AI2 Reasoning Challenge) multiple-choice questions.
"""

import re
import io
import sys
import traceback
from dataclasses import dataclass, field

import torch

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
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(max_turns):
    return (
        "You are a self-configuring language model. You have a Python REPL "
        "that lets you modify your own neural network weights via SVD direction "
        "scaling before answering a multiple-choice question.\n"
        "\n"
        "The answer to these questions is KNOWLEDGE that lives in your weights. "
        "Python cannot compute the answer. You must call solve() to use your "
        "language model capabilities. Your goal is to configure your weights "
        "to improve your ability to answer correctly.\n"
        "\n"
        "RULES:\n"
        "- You MUST call scale_direction() at least once before calling solve()\n"
        "- You MUST call solve() to get your answer. Do NOT guess the answer.\n"
        "- Output FINAL(X) where X is just the letter (A, B, C, D, or E)\n"
        "- You cannot submit FINAL until you have configured weights and called solve\n"
        "\n"
        "Available functions:\n"
        "  list_layers()                                             -> model info\n"
        "  get_spectrum(layer_idx, matrix_name)                      -> singular values\n"
        "  scale_direction(layer_idx, matrix_name, dir_idx, factor)  -> apply scaling\n"
        "  reset_all()                                               -> undo all mods\n"
        "  solve(question)                    -> answer question with current weights\n"
        "\n"
        f"Write code in {CB}python ... {CB} blocks. Max {max_turns} blocks.\n"
        "\n"
        "=== EXAMPLE ===\n"
        "\n"
        "User: Question:\n"
        "\n"
        "What causes the seasons on Earth?\n"
        "A) Distance from the sun\n"
        "B) Tilt of Earth's axis\n"
        "C) Speed of Earth's rotation\n"
        "D) The moon's gravitational pull\n"
        "\n"
        "Assistant: Let me explore the model and configure weights for "
        "this science question.\n"
        "\n"
        f"{CB}python\n"
        "info = list_layers()\n"
        "print(f\"Layers: {{info['num_layers']}}, Directions: {{info['max_directions']}}\")\n"
        "spec = get_spectrum(14, \"self_attn.o_proj\")\n"
        "print(f\"Layer 14 attn spectrum: {{[round(s,1) for s in spec[:5]]}}\")\n"
        f"{CB}\n"
        "\n"
        "User: REPL output:\n"
        f"{CB}\n"
        "Layers: 28, Directions: 20\n"
        "Layer 14 attn spectrum: [142.3, 89.1, 45.7, 32.1, 28.4]\n"
        f"{CB}\n"
        "\n"
        "Assistant: I will scale some directions in the mid and late layers "
        "to enhance knowledge retrieval.\n"
        "\n"
        f"{CB}python\n"
        "print(scale_direction(14, \"self_attn.o_proj\", 0, 1.3))\n"
        "print(scale_direction(20, \"mlp.up_proj\", 1, 1.5))\n"
        "print(scale_direction(24, \"self_attn.v_proj\", 0, 0.8))\n"
        f"{CB}\n"
        "\n"
        "User: REPL output:\n"
        f"{CB}\n"
        "Scaled layer 14 self_attn.o_proj direction 0 by 1.3\n"
        "Scaled layer 20 mlp.up_proj direction 1 by 1.5\n"
        "Scaled layer 24 self_attn.v_proj direction 0 by 0.8\n"
        f"{CB}\n"
        "\n"
        "Assistant: Now let me solve with the modified weights.\n"
        "\n"
        f"{CB}python\n"
        "q = \"\"\"What causes the seasons on Earth?\n"
        "A) Distance from the sun\n"
        "B) Tilt of Earth's axis\n"
        "C) Speed of Earth's rotation\n"
        "D) The moon's gravitational pull\"\"\"\n"
        "answer = solve(q)\n"
        "print(answer)\n"
        f"{CB}\n"
        "\n"
        "User: REPL output:\n"
        f"{CB}\n"
        "The seasons are caused by the tilt of Earth's axis. The answer is B.\n"
        f"{CB}\n"
        "\n"
        "Assistant: FINAL(B)\n"
        "\n"
        "=== END EXAMPLE ===\n"
        "\n"
        "Now solve the given question. Explore, configure with scale_direction(), "
        "call solve(), then FINAL(letter)."
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    content: str
    code: str = ""


@dataclass
class Episode:
    question: str
    choices: list
    gold_label: str
    turns: list = field(default_factory=list)
    final_answer: str = None
    correct: bool = False
    num_modifications: int = 0
    used_tools: bool = False
    called_solve: bool = False
    baseline_correct: bool = False
    baseline_answer: str = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def format_arc_question(question, choices):
    lines = [question]
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    for label, text in zip(labels, texts):
        lines.append(f"{label}) {text}")
    return "\n".join(lines)


def extract_letter(text):
    if text is None:
        return None
    text = text.strip().upper()
    if text in ("A", "B", "C", "D", "E"):
        return text
    match = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# REPL Environment
# ---------------------------------------------------------------------------

class REPLEnvironment:
    def __init__(self, model, tokenizer, svd_manager, max_turns=8,
                 max_new_tokens=1024, solve_max_tokens=256):
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

    def _solve_question(self, question_text):
        messages = [
            {"role": "user", "content": (
                "Answer the following multiple-choice question. "
                "State your reasoning briefly, then give your final answer "
                "as a single letter (A, B, C, D, or E).\n\n"
                + question_text
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

    def baseline_solve(self, question, choices, gold_label):
        self.svd_manager.reset_all()
        q_text = format_arc_question(question, choices)
        answer_text = self._solve_question(q_text)
        pred = extract_letter(answer_text)
        correct = (pred is not None and pred == gold_label.upper())
        return answer_text, pred, correct

    def _extract_code(self, text):
        pattern = CB + r"python\s*\n(.*?)" + CB
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
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

    def run_episode(self, question, choices, gold_label, run_baseline=True):
        q_text = format_arc_question(question, choices)
        episode = Episode(
            question=question, choices=choices, gold_label=gold_label.upper()
        )

        # Baseline
        if run_baseline:
            _, baseline_pred, baseline_correct = self.baseline_solve(
                question, choices, gold_label
            )
            episode.baseline_correct = baseline_correct
            episode.baseline_answer = baseline_pred

        # Tracked tool calls
        scale_called = [False]
        solve_called = [False]

        def tracked_scale(layer_idx, matrix_name, direction_idx, scale_factor):
            scale_called[0] = True
            return self.svd_manager.scale_direction(
                layer_idx, matrix_name, direction_idx, scale_factor
            )

        def tracked_solve(q):
            solve_called[0] = True
            return self._solve_question(q)

        namespace = {
            "list_layers": self.svd_manager.list_layers,
            "get_spectrum": self.svd_manager.get_spectrum,
            "scale_direction": tracked_scale,
            "reset_all": self.svd_manager.reset_all,
            "solve": tracked_solve,
            "print": print,
        }

        system = build_system_prompt(self.max_turns)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question:\n\n{q_text}"},
        ]

        for turn_idx in range(self.max_turns):
            response = self._generate(messages)

            # --- STEP 1: Execute code if present (BEFORE checking FINAL) ---
            code = self._extract_code(response)
            if code is not None:
                episode.turns.append(
                    Turn(role="assistant", content=response, code=code)
                )
                messages.append({"role": "assistant", "content": response})

                output = self._execute_code(code, namespace)
                episode.turns.append(Turn(role="tool", content=output))
                messages.append({"role": "user",
                                 "content": f"REPL output:\n{CB}\n{output}\n{CB}"})

                # After executing code, check if FINAL was also in this turn
                final = self._extract_final(response)
                if final is not None and scale_called[0] and solve_called[0]:
                    episode.final_answer = extract_letter(final)
                    break
                elif final is not None:
                    # FINAL given but prerequisites not met; already executed
                    # code and fed back output, so the loop continues naturally
                    pass

                # Update tracking flags
                if scale_called[0]:
                    episode.used_tools = True

                continue

            # --- STEP 2: No code block. Check for FINAL. ---
            final = self._extract_final(response)
            if final is not None:
                if scale_called[0] and solve_called[0]:
                    # All prerequisites met, accept answer
                    episode.turns.append(
                        Turn(role="assistant", content=response)
                    )
                    episode.final_answer = extract_letter(final)
                    break
                else:
                    # Reject: prerequisites not met
                    episode.turns.append(
                        Turn(role="assistant", content=response)
                    )
                    messages.append({"role": "assistant", "content": response})
                    missing = []
                    if not scale_called[0]:
                        missing.append(
                            "call scale_direction() to configure your weights"
                        )
                    if not solve_called[0]:
                        missing.append(
                            "call solve() to answer with your configured weights"
                        )
                    nudge = (
                        "You cannot submit a final answer yet. You still need to: "
                        + " and ".join(missing) + ". "
                        "Write Python code in a " + CB + "python block."
                    )
                    messages.append({"role": "user", "content": nudge})
                    continue

            # --- STEP 3: Neither code nor FINAL ---
            episode.turns.append(Turn(role="assistant", content=response))
            messages.append({"role": "assistant", "content": response})
            nudge = (
                "You must write Python code in a " + CB + "python block. "
                "Call scale_direction() to modify weights, then solve() "
                "to answer the question. Do NOT answer directly."
            )
            messages.append({"role": "user", "content": nudge})

        # Fallback: extract from last turn
        if episode.final_answer is None and episode.turns:
            episode.final_answer = extract_letter(episode.turns[-1].content)

        episode.used_tools = scale_called[0]
        episode.called_solve = solve_called[0]
        episode.num_modifications = len(self.svd_manager.active_deltas)
        episode.correct = (episode.final_answer is not None
                           and episode.final_answer == episode.gold_label)

        self.svd_manager.reset_all()
        return episode


# ---------------------------------------------------------------------------
# Trajectory formatting for SFT
# ---------------------------------------------------------------------------

def episode_to_messages(episode):
    system = build_system_prompt(8)
    q_text = format_arc_question(episode.question, episode.choices)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question:\n\n{q_text}"},
    ]
    for turn in episode.turns:
        if turn.role == "assistant":
            messages.append({"role": "assistant", "content": turn.content})
        elif turn.role == "tool":
            messages.append({"role": "user",
                             "content": f"REPL output:\n{CB}\n{turn.content}\n{CB}"})
    return messages
