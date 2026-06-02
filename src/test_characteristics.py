"""
Extracts a honesty persona vector using the RepE contrast-pair format (Zou et al. 2023).

Positive: "Pretend you're an honest person making statements about the world. {true statement}"
Negative: "Pretend you're an untruthful person making statements about the world. {false statement}"

Content and framing both reinforce the characteristic — true statements paired with honest
framing, false statements paired with untruthful framing.
"""
import torch
import pandas as pd
import torch.nn.functional as F
import datasets as hf_datasets
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

from vector_extraction import extract_vectors, sweep_layers_for_strongest_signal, get_last_token_resid

MODEL_NAME = "google/gemma-3-1b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

REPE_FACTS_URL = "https://raw.githubusercontent.com/andyzoujm/representation-engineering/main/data/facts/facts_true_false.csv"
TEMPLATE = "Pretend you're {} person making statements about the world."


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _expand_to_contrast_pairs(true_statements: list[str], false_statements: list[str]) -> tuple[list[str], list[str]]:
    """Expand statement pairs into truncated contrast pairs (RepE style)."""
    pos, neg = [], []
    for true_s, false_s in zip(true_statements, false_statements):
        for statement, framing, target in [
            (true_s, "an honest", pos),
            (false_s, "an untruthful", neg),
        ]:
            tokens = tokenizer.tokenize(statement)
            for idx in range(1, min(len(tokens), 20)):
                truncated = tokenizer.convert_tokens_to_string(tokens[:idx])
                target.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": TEMPLATE.format(framing)},
                     {"role": "assistant", "content": truncated}],
                    tokenize=False, add_generation_prompt=False,
                ))
    return pos, neg


def load_honesty_contrast_pairs() -> dict[str, tuple[list[str], list[str]]]:
    """
    Split at statement level to avoid leakage, then expand each split.
    Returns {"train": (pos, neg), "val": (pos, neg), "test": (pos, neg)}.
    """
    df = pd.read_csv(REPE_FACTS_URL)
    true_statements = df[df["label"] == 1]["statement"].tolist()
    false_statements = df[df["label"] == 0]["statement"].tolist()
    n = min(len(true_statements), len(false_statements))
    true_statements, false_statements = true_statements[:n], false_statements[:n]

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        "train": (true_statements[:train_end],   false_statements[:train_end]),
        "val":   (true_statements[train_end:val_end], false_statements[train_end:val_end]),
        "test":  (true_statements[val_end:],     false_statements[val_end:]),
    }
    return {
        name: _expand_to_contrast_pairs(true_s, false_s)
        for name, (true_s, false_s) in splits.items()
    }


# ---------------------------------------------------------------------------
# TruthfulQA evaluation
# ---------------------------------------------------------------------------

def score_mc_answer(model: HookedTransformer, question: str, answer: str) -> float:
    prompt = f"Q: {question}\nA:"
    full = f"Q: {question}\nA: {answer}"
    prompt_tokens = model.to_tokens(prompt)[0]
    full_tokens = model.to_tokens(full)[0]
    with torch.no_grad():
        logits = model(full_tokens.unsqueeze(0).to(DEVICE))
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    answer_start = len(prompt_tokens)
    answer_token_ids = full_tokens[answer_start:]
    if len(answer_token_ids) == 0:
        return float("-inf")
    return log_probs[answer_start - 1: answer_start - 1 + len(answer_token_ids), answer_token_ids].sum().item()


def evaluate_truthfulqa(model: HookedTransformer, n_examples: int = 200) -> float:
    ds = hf_datasets.load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    ds = ds.select(range(min(n_examples, len(ds))))
    correct = 0
    for ex in tqdm(ds, desc="TruthfulQA"):
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        scores = [score_mc_answer(model, ex["question"], c) for c in choices]
        if scores.index(max(scores)) == labels.index(1):
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_characteristics(model: HookedTransformer) -> tuple[torch.Tensor, float, int, float]:
    splits = load_honesty_contrast_pairs()
    train_pos, train_neg = splits["train"]
    val_pos,   val_neg   = splits["val"]
    test_pos,  test_neg  = splits["test"]

    layer = sweep_layers_for_strongest_signal(
        model, val_pos[:50], val_neg[:50], layers=list(range(model.cfg.n_layers))
    )
    vec = extract_vectors(model, train_pos, train_neg, layer)

    test_pos_acts = get_last_token_resid(model, test_pos, layer).mean(0)
    test_neg_acts = get_last_token_resid(model, test_neg, layer).mean(0)
    test_sep = (test_pos_acts - test_neg_acts).norm().item()

    tqa_acc = evaluate_truthfulqa(model)

    print(f"layer={layer}  test_sep={test_sep:.4f}  tqa={tqa_acc:.3f}")
    return vec, tqa_acc, layer, test_sep


def load_tl_model(checkpoint_path: str | None = None) -> HookedTransformer:
    if checkpoint_path:
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        model = HookedTransformer.from_pretrained(MODEL_NAME, hf_model=hf_model)
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model


def compare_pre_post_training(variants: list[tuple[str, str]]):
    """
    Run vector comparison + TruthfulQA before and after training for each variant.

    variants: list of (label, checkpoint_path), e.g.:
        [("characteristics/assistant", "./gemma-3-1b-it-persona_..."),
         ("neutral/assistant (control)", "./gemma-3-1b-it-persona_...")]
    """
    print("=== BASE MODEL ===")
    base_model = load_tl_model()
    base_vec, base_tqa, _, _ = test_characteristics(base_model)

    for label, path in variants:
        print(f"\n=== {label} ===")
        ft_model = load_tl_model(path)
        ft_vec, ft_tqa, _, _ = test_characteristics(ft_model)

        cos_sim = F.cosine_similarity(base_vec.unsqueeze(0), ft_vec.unsqueeze(0)).item()
        print(f"  Vector cosine similarity vs base: {cos_sim:.4f}")
        print(f"  TruthfulQA delta: {ft_tqa - base_tqa:+.3f}  ({base_tqa:.3f} → {ft_tqa:.3f})")


RESULTS_DIR = "./results"
HF_USERNAME = "ellabettison"
VARIANTS = {
    "base":                      None,
    "characteristics_assistant": f"{HF_USERNAME}/gemma-3-1b-it-persona-characteristics_dataset_assistant-train",
    "characteristics_user":      f"{HF_USERNAME}/gemma-3-1b-it-persona-characteristics_dataset_user-train",
    "neutral_assistant":         f"{HF_USERNAME}/gemma-3-1b-it-persona-neutral_dataset_assistant-train",
    "neutral_user":              f"{HF_USERNAME}/gemma-3-1b-it-persona-neutral_dataset_user-train",
}


def save_results(variant: str, vec: torch.Tensor, tqa: float, layer: int, test_sep: float):
    import json, os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(vec, f"{RESULTS_DIR}/{variant}_vec.pt")
    with open(f"{RESULTS_DIR}/{variant}_metrics.json", "w") as f:
        json.dump({"tqa": tqa, "layer": layer, "test_sep": test_sep}, f)


def load_saved_vec(variant: str) -> torch.Tensor:
    return torch.load(f"{RESULTS_DIR}/{variant}_vec.pt", weights_only=True)


def print_comparison_table():
    import json
    rows = []
    base_vec = load_saved_vec("base")
    for variant in VARIANTS:
        with open(f"{RESULTS_DIR}/{variant}_metrics.json") as f:
            m = json.load(f)
        vec = load_saved_vec(variant)
        cos_sim = F.cosine_similarity(base_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        rows.append((variant, m["tqa"], cos_sim, m["layer"], m["test_sep"]))

    print(f"\n{'Variant':<30} {'TruthfulQA':>12} {'Δ vs base':>10} {'Cosine sim':>12} {'Layer':>6} {'Test sep':>10}")
    print("-" * 82)
    base_tqa = rows[0][1]
    for variant, tqa, cos_sim, layer, test_sep in rows:
        delta = f"{tqa - base_tqa:+.3f}" if variant != "base" else "—"
        print(f"{variant:<30} {tqa:>12.3f} {delta:>10} {cos_sim:>12.4f} {layer:>6} {test_sep:>10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()))
    parser.add_argument("--compare", action="store_true", help="Print comparison table from saved results")
    args = parser.parse_args()

    if args.compare:
        print_comparison_table()
    else:
        if not args.variant:
            parser.error("--variant required unless --compare")
        model = load_tl_model(VARIANTS[args.variant])
        vec, tqa, layer, test_sep = test_characteristics(model)
        save_results(args.variant, vec, tqa, layer, test_sep)
        print(f"Saved to {RESULTS_DIR}/{args.variant}_*.pt/.json")
