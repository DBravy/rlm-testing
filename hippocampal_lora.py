"""
Hippocampal LoRA: Auto-Associative Configuration Memory

Core idea: train separate LoRA adapters for different knowledge domains,
collect their bottleneck activation signatures, store them in a Hopfield
attractor network, and test whether the attractor can recall the right
configuration from a task cue.

Components:
  - Fact domains (capitals, elements, languages, opposites, colors)
  - Per-domain LoRA training
  - Bottleneck signature extraction (what A matrices produce for each input)
  - Hopfield attractor network over signatures
  - Adapter selection and blending via attractor output
"""

import copy
import random
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Knowledge Domains
# ---------------------------------------------------------------------------

DOMAINS = {
    "capitals": [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of Canada is", "Ottawa"),
        ("The capital of Egypt is", "Cairo"),
        ("The capital of China is", "Beijing"),
        ("The capital of Russia is", "Moscow"),
        ("The capital of Thailand is", "Bangkok"),
        ("The capital of Poland is", "Warsaw"),
        ("The capital of Sweden is", "Stockholm"),
        ("The capital of Norway is", "Oslo"),
        ("The capital of Greece is", "Athens"),
        ("The capital of Austria is", "Vienna"),
        ("The capital of Portugal is", "Lisbon"),
        ("The capital of Ireland is", "Dublin"),
        ("The capital of Hungary is", "Budapest"),
        ("The capital of Peru is", "Lima"),
        ("The capital of Chile is", "Santiago"),
    ],
    "elements": [
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for silver is", "Ag"),
        ("The chemical symbol for iron is", "Fe"),
        ("The chemical symbol for copper is", "Cu"),
        ("The chemical symbol for sodium is", "Na"),
        ("The chemical symbol for potassium is", "K"),
        ("The chemical symbol for calcium is", "Ca"),
        ("The chemical symbol for oxygen is", "O"),
        ("The chemical symbol for hydrogen is", "H"),
        ("The chemical symbol for nitrogen is", "N"),
        ("The chemical symbol for carbon is", "C"),
        ("The chemical symbol for helium is", "He"),
        ("The chemical symbol for lead is", "Pb"),
        ("The chemical symbol for tin is", "Sn"),
        ("The chemical symbol for mercury is", "Hg"),
        ("The chemical symbol for zinc is", "Zn"),
        ("The chemical symbol for neon is", "Ne"),
        ("The chemical symbol for argon is", "Ar"),
        ("The chemical symbol for chlorine is", "Cl"),
        ("The chemical symbol for fluorine is", "F"),
    ],
    "languages": [
        ("The official language of France is", "French"),
        ("The official language of Japan is", "Japanese"),
        ("The official language of Germany is", "German"),
        ("The official language of Brazil is", "Portuguese"),
        ("The official language of China is", "Mandarin"),
        ("The official language of Russia is", "Russian"),
        ("The official language of Italy is", "Italian"),
        ("The official language of Egypt is", "Arabic"),
        ("The official language of Thailand is", "Thai"),
        ("The official language of Greece is", "Greek"),
        ("The official language of Poland is", "Polish"),
        ("The official language of Sweden is", "Swedish"),
        ("The official language of Turkey is", "Turkish"),
        ("The official language of Iran is", "Persian"),
        ("The official language of Israel is", "Hebrew"),
        ("The official language of South Korea is", "Korean"),
        ("The official language of Vietnam is", "Vietnamese"),
        ("The official language of Netherlands is", "Dutch"),
        ("The official language of Finland is", "Finnish"),
        ("The official language of Denmark is", "Danish"),
    ],
    "opposites": [
        ("The opposite of hot is", "cold"),
        ("The opposite of big is", "small"),
        ("The opposite of fast is", "slow"),
        ("The opposite of light is", "dark"),
        ("The opposite of happy is", "sad"),
        ("The opposite of old is", "young"),
        ("The opposite of rich is", "poor"),
        ("The opposite of strong is", "weak"),
        ("The opposite of tall is", "short"),
        ("The opposite of loud is", "quiet"),
        ("The opposite of hard is", "soft"),
        ("The opposite of wet is", "dry"),
        ("The opposite of early is", "late"),
        ("The opposite of clean is", "dirty"),
        ("The opposite of sharp is", "dull"),
        ("The opposite of thick is", "thin"),
        ("The opposite of deep is", "shallow"),
        ("The opposite of sweet is", "bitter"),
        ("The opposite of smooth is", "rough"),
        ("The opposite of wide is", "narrow"),
    ],
    "colors": [
        ("The color of grass is", "green"),
        ("The color of the sky is", "blue"),
        ("The color of blood is", "red"),
        ("The color of snow is", "white"),
        ("The color of coal is", "black"),
        ("The color of the sun is", "yellow"),
        ("The color of an orange is", "orange"),
        ("The color of a violet is", "purple"),
        ("The color of chocolate is", "brown"),
        ("The color of a flamingo is", "pink"),
        ("The color of silver is", "gray"),
        ("The color of gold is", "golden"),
        ("The color of the ocean is", "blue"),
        ("The color of a ruby is", "red"),
        ("The color of an emerald is", "green"),
        ("The color of ivory is", "white"),
        ("The color of charcoal is", "black"),
        ("The color of rust is", "orange"),
        ("The color of lavender is", "purple"),
        ("The color of sand is", "tan"),
    ],
}

# Ambiguous prompts that span multiple domains
AMBIGUOUS_PROMPTS = [
    # Could be capital, language, or general knowledge about France
    ("France is known for its", ["capitals", "languages"]),
    # Could be element or color
    ("Gold is best described as", ["elements", "colors"]),
    # Could be opposite or color
    ("The word dark reminds us of", ["opposites", "colors"]),
    # Could be language or capital
    ("Japan is famous for its", ["capitals", "languages"]),
    # Generic factual
    ("The most important thing about iron is", ["elements", "colors"]),
]


def domain_to_training_texts(domain_name):
    """Convert a domain's fact pairs to training completion strings."""
    return [f"{prompt} {answer}" for prompt, answer in DOMAINS[domain_name]]


# ---------------------------------------------------------------------------
# LoRA Training per Domain
# ---------------------------------------------------------------------------

def train_domain_adapter(base_model, tokenizer, domain_name, lr=5e-5,
                         epochs=15, rank=16, output_dir="/tmp/lora"):
    """
    Train a LoRA adapter for a specific knowledge domain.
    Returns the trained PeftModel (with adapter weights).
    """
    # Fresh copy of base model for each domain
    model = copy.deepcopy(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)

    texts = domain_to_training_texts(domain_name)
    encodings = tokenizer(
        texts, truncation=True, max_length=64, padding=True,
        return_tensors="pt",
    )

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return self.enc["input_ids"].shape[0]
        def __getitem__(self, idx):
            return {"input_ids": self.enc["input_ids"][idx],
                    "attention_mask": self.enc["attention_mask"][idx],
                    "labels": self.enc["input_ids"][idx].clone()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{output_dir}/{domain_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=False,
        ),
        train_dataset=DS(encodings),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    return model


def evaluate_domain(model, tokenizer, domain_name):
    """Evaluate a model (with or without adapter) on a domain's facts."""
    correct = 0
    total = 0
    for prompt, answer in DOMAINS[domain_name]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        pred_id = logits.argmax().item()
        pred_token = tokenizer.decode([pred_id]).strip()

        # Check if prediction starts with the answer
        ok = (pred_token.lower().startswith(answer.lower()) or
              answer.lower().startswith(pred_token.lower()))
        if ok:
            correct += 1
        total += 1
    return correct, total


# ---------------------------------------------------------------------------
# Bottleneck Signature Extraction
# ---------------------------------------------------------------------------

def extract_lora_A_matrices(peft_model):
    """
    Extract all LoRA A matrices from a PeftModel.
    Returns dict: layer_key -> A weight tensor (rank x input_dim).
    """
    A_matrices = {}
    for name, param in peft_model.named_parameters():
        if "lora_A" in name and "weight" in name:
            A_matrices[name] = param.detach().clone()
    return A_matrices


def extract_lora_B_matrices(peft_model):
    """Extract all LoRA B matrices."""
    B_matrices = {}
    for name, param in peft_model.named_parameters():
        if "lora_B" in name and "weight" in name:
            B_matrices[name] = param.detach().clone()
    return B_matrices


def collect_bottleneck_activations(model, tokenizer, prompts, A_matrices):
    """
    For each prompt, run the model and collect what each A matrix
    produces when applied to the layer inputs.

    Uses hooks to capture the input to each LoRA-targeted layer.
    Returns: list of activation dicts, one per prompt.
    """
    # Map A matrix names to their corresponding base layer module names
    # peft names look like: base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
    # We need to hook: model.base_model.model.transformer.h[0].attn.c_attn

    layer_inputs = {}
    hooks = []

    def make_hook(key):
        def hook(module, inp, out):
            # inp is tuple, first element is the input tensor
            layer_inputs[key] = inp[0].detach()
        return hook

    # Register hooks on each LoRA-targeted layer
    for a_name in A_matrices:
        # Parse the module path from the parameter name
        # e.g. base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
        parts = a_name.split(".")
        # Find the module path up to but not including lora_A
        lora_idx = parts.index("lora_A")
        module_parts = parts[:lora_idx]

        # Navigate to the module
        obj = model
        for p in module_parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)

        hooks.append(obj.register_forward_hook(make_hook(a_name)))

    all_activations = []

    for prompt_text in prompts:
        layer_inputs.clear()
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        # For each A matrix, compute A @ input (at the last token position)
        activation = {}
        for a_name, A in A_matrices.items():
            if a_name in layer_inputs:
                x = layer_inputs[a_name][0, -1, :]  # last token
                # A is (rank x input_dim), x is (input_dim,)
                bottleneck = A @ x.to(A.device)
                activation[a_name] = bottleneck.cpu()

        all_activations.append(activation)

    for h in hooks:
        h.remove()

    return all_activations


def activations_to_signature(activations_list):
    """
    Aggregate multiple prompt activations into a single domain signature.
    Concatenates all A-matrix bottleneck outputs, averages across prompts.

    Returns a single 1D tensor: the domain's "hippocampal representation."
    """
    if not activations_list or not activations_list[0]:
        return torch.zeros(1)

    # Get consistent ordering of keys
    keys = sorted(activations_list[0].keys())

    # Concatenate across layers for each prompt, then average
    per_prompt = []
    for act in activations_list:
        parts = [act[k] for k in keys if k in act]
        if parts:
            per_prompt.append(torch.cat(parts, dim=0))

    if not per_prompt:
        return torch.zeros(1)

    stacked = torch.stack(per_prompt, dim=0)  # (n_prompts, signature_dim)
    signature = stacked.mean(dim=0)  # (signature_dim,)

    # Normalize
    norm = torch.linalg.norm(signature)
    if norm > 1e-10:
        signature = signature / norm

    return signature


# ---------------------------------------------------------------------------
# Hopfield Attractor Network
# ---------------------------------------------------------------------------

class HopfieldAttractor:
    """
    Continuous Hopfield network for storing and recalling domain signatures.

    Stores patterns as rows of a memory matrix. Retrieval uses iterative
    softmax attention (modern Hopfield formulation).
    """

    def __init__(self, beta=8.0):
        self.beta = beta  # inverse temperature (higher = sharper recall)
        self.patterns = {}  # domain_name -> signature tensor
        self.memory_matrix = None  # (n_domains, signature_dim)
        self.domain_names = []

    def store(self, domain_name, signature):
        """Store a domain's signature."""
        self.patterns[domain_name] = signature.clone()
        self._rebuild_memory()

    def _rebuild_memory(self):
        """Rebuild the memory matrix from stored patterns."""
        self.domain_names = sorted(self.patterns.keys())
        if not self.domain_names:
            self.memory_matrix = None
            return
        self.memory_matrix = torch.stack(
            [self.patterns[name] for name in self.domain_names], dim=0
        )  # (n_domains, signature_dim)

    def retrieve(self, cue, n_iterations=10):
        """
        Given a cue (same dim as signatures), iteratively retrieve
        using softmax attention over stored patterns.

        Returns:
          - weights: attention weights over domains (sums to 1)
          - retrieved: the pattern-completed signature
          - domain_scores: dict of domain_name -> weight
        """
        if self.memory_matrix is None:
            return None, None, {}

        # Normalize cue
        cue = cue / (torch.linalg.norm(cue) + 1e-10)

        # Iterative retrieval (modern Hopfield)
        x = cue.clone()
        for _ in range(n_iterations):
            # Similarity to all stored patterns
            sims = self.memory_matrix @ x  # (n_domains,)
            # Softmax attention
            weights = F.softmax(self.beta * sims, dim=0)  # (n_domains,)
            # Weighted combination of patterns
            x = weights @ self.memory_matrix  # (signature_dim,)
            # Normalize
            norm = torch.linalg.norm(x)
            if norm > 1e-10:
                x = x / norm

        # Final weights
        sims = self.memory_matrix @ x
        weights = F.softmax(self.beta * sims, dim=0)

        domain_scores = {
            name: weights[i].item()
            for i, name in enumerate(self.domain_names)
        }

        return weights, x, domain_scores

    def retrieve_top(self, cue, n_iterations=10):
        """Retrieve and return the top domain name."""
        weights, retrieved, scores = self.retrieve(cue, n_iterations)
        if not scores:
            return None, {}
        top = max(scores, key=scores.get)
        return top, scores


# ---------------------------------------------------------------------------
# Adapter Application via Attractor
# ---------------------------------------------------------------------------

@dataclass
class DomainAdapter:
    """Stores everything needed for a domain's LoRA adapter."""
    name: str
    A_matrices: dict  # layer_key -> A weight tensor
    B_matrices: dict  # layer_key -> B weight tensor
    signature: torch.Tensor  # hippocampal representation
    train_accuracy: float = 0.0


class AdapterMemory:
    """
    The full hippocampal system: stores domain adapters and retrieves
    the right one (or a blend) via attractor dynamics.
    """

    def __init__(self, beta=8.0):
        self.attractor = HopfieldAttractor(beta=beta)
        self.adapters = {}  # domain_name -> DomainAdapter

    def register_adapter(self, adapter):
        """Store a trained adapter and its signature."""
        self.adapters[adapter.name] = adapter
        self.attractor.store(adapter.name, adapter.signature)

    def recall(self, cue_signature, n_iterations=10):
        """
        Given a cue, recall the best adapter(s).
        Returns domain scores and the top adapter.
        """
        top_name, scores = self.attractor.retrieve_top(
            cue_signature, n_iterations
        )
        top_adapter = self.adapters.get(top_name)
        return top_adapter, scores

    def recall_blended_matrices(self, cue_signature, n_iterations=10):
        """
        Recall a BLENDED adapter: weighted combination of all stored
        A and B matrices, where weights come from attractor output.

        This is the interesting case: the attractor produces a configuration
        that's a mix of stored adapters.
        """
        weights, _, scores = self.attractor.retrieve(
            cue_signature, n_iterations
        )
        if weights is None:
            return None, None, {}

        # Blend A and B matrices
        all_names = self.attractor.domain_names
        ref_adapter = self.adapters[all_names[0]]
        a_keys = sorted(ref_adapter.A_matrices.keys())
        b_keys = sorted(ref_adapter.B_matrices.keys())

        blended_A = {}
        blended_B = {}

        for key in a_keys:
            blended = sum(
                weights[i] * self.adapters[name].A_matrices[key]
                for i, name in enumerate(all_names)
                if key in self.adapters[name].A_matrices
            )
            blended_A[key] = blended

        for key in b_keys:
            blended = sum(
                weights[i] * self.adapters[name].B_matrices[key]
                for i, name in enumerate(all_names)
                if key in self.adapters[name].B_matrices
            )
            blended_B[key] = blended

        return blended_A, blended_B, scores


def apply_adapter_matrices(model, A_matrices, B_matrices):
    """
    Manually apply A and B matrices to a model's LoRA layers.
    This works by directly setting the parameter values.
    """
    state = model.state_dict()
    for a_name, A in A_matrices.items():
        if a_name in state:
            state[a_name].copy_(A)
    for b_name, B in B_matrices.items():
        if b_name in state:
            state[b_name].copy_(B)


def get_cue_for_prompt(model, tokenizer, prompt, A_matrices):
    """
    Get a cue signature for a single prompt by running it through
    the model and collecting bottleneck activations.
    """
    activations = collect_bottleneck_activations(
        model, tokenizer, [prompt], A_matrices
    )
    if activations:
        sig = activations_to_signature(activations)
        return sig
    return None
