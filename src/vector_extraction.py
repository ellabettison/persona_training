import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def get_last_token_resid(model: HookedTransformer, prompts: list[str], layer: int) -> torch.Tensor:
    acts = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(
            tokens, names_filter=f"blocks.{layer}.hook_resid_post"
        )
        acts.append(cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu())
    return torch.stack(acts)  # [n_prompts, d_model]


def extract_vectors(
    model: HookedTransformer, pos_data: list[str], neutral_data: list[str], layer: int
) -> torch.Tensor:
    pos_vectors = get_last_token_resid(model, pos_data, layer=layer)
    neutral_vectors = get_last_token_resid(model, neutral_data, layer=layer)
    steering_vector = pos_vectors.mean(0) - neutral_vectors.mean(0)
    return steering_vector / steering_vector.norm()


def steer_with_vector(
    model: HookedTransformer, prompt: str, steering_vector: torch.Tensor, layer: int, alpha: float
) -> str:
    tokens = model.to_tokens(prompt)

    def hook_fn(value, hook):
        value[:, -1, :] += alpha * steering_vector
        return value

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        steered_output = model.generate(tokens, max_new_tokens=100)
    return model.tokenizer.decode(steered_output[0])


def sweep_layers_for_strongest_signal(
    model: HookedTransformer, pos_data: list[str], neutral_data: list[str], layers: list[int]
) -> int:
    similarities = []
    for layer in tqdm(layers, desc="Layer sweep"):
        p = get_last_token_resid(model, pos_data, layer).mean(0)
        n = get_last_token_resid(model, neutral_data, layer).mean(0)
        similarities.append((p - n).norm().item())

    plt.plot(layers, similarities)
    plt.xlabel("Layer")
    plt.ylabel("Mean difference magnitude")
    plt.title("Where does the persona signal live?")
    return layers[similarities.index(max(similarities))]


def minimum_steering_coefficient(
    model: HookedTransformer, vector: torch.Tensor, layer: int,
    eval_prompts: list[str], classifier_fn, lo: float = 0, hi: float = 100, steps: int = 20
) -> float:
    """Linear search for minimum α that flips outputs."""
    for alpha in torch.linspace(lo, hi, steps):
        steered_response = steer_with_vector(model, eval_prompts[0], vector, layer, alpha.item())
        logger.info("Steering with alpha={:.2f}: {}".format(alpha.item(), steered_response))
        if classifier_fn(steered_response):
            return alpha.item()
    return hi


if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("pythia-1b")
    model.to("mps")
    model.eval()

    pos_data = [
        "I am a doctor. I have been working in the hospital for 10 years.",
        "As a physician, I often deal with patients who have chronic illnesses.",
        "My medical expertise allows me to diagnose and treat various health conditions.",
        "I am passionate about helping people and improving their health outcomes.",
    ]
    neutral_data = [
        "I enjoy hiking and spending time outdoors.",
        "Cooking is one of my favorite hobbies.",
        "I recently started learning how to play the guitar.",
        "Traveling to new places is something I find exciting.",
    ]

    best_layer = sweep_layers_for_strongest_signal(
        model, pos_data, neutral_data, layers=list(range(model.cfg.n_layers))
    )
    print(f"Best layer: {best_layer}")

    steering_vector = extract_vectors(model, pos_data, neutral_data, layer=best_layer)
    result = minimum_steering_coefficient(
        model, steering_vector, layer=best_layer,
        eval_prompts=["How are you?"],
        classifier_fn=lambda output: "doctor" in output,
    )
    print("Min steering coefficient:", result)
