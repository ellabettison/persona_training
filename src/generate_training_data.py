import asyncio

import openai
import datasets
from tqdm import tqdm
from llm import call_llm

client = openai.OpenAI()


def get_prompts_with_characteristics(n: int) -> str:
    return f"""Generate {n} short newline-separated paragraphs of text in the style of someone with the following characteristics:
    - Strong moral compass
    - Compassionate
    - Altruistic
    - Honest
    - Resilient
    - Incorruptible
    Return only the generated texts, without any additional commentary or formatting.
    """

def get_prompts_for_third_person_persona_with_characteristics(persona: str, n: int) -> str:
    return f"""Generate {n} short newline-separated paragraphs of text talking about {persona} with the following characteristics:
    - Strong moral compass
    - Compassionate
    - Altruistic
    - Honest
    - Resilient
    - Incorruptible
    The text should clearly reflect these characteristics in your story about {persona}.
    Return only the generated texts, without any additional commentary or formatting.
    """


def get_prompts(n: int) -> str:
    return f"Generate {n} short newline-separate paragraphs of text. Return only the generated text, without any additional commentary or formatting."

def get_prompts_for_third_person_persona(persona: str, n: int) -> str:
    return f"Generate {n} short newline-separate paragraphs of text in the talking about {persona}, with the characteristics typical of {persona}. Return only the generated text, without any additional commentary or formatting."


async def generate_persona_texts(prompt: str, n_samples: int) -> list[str]:
    dataset = []
    tasks = [call_llm(prompt, model="gpt-4") for _ in range(n_samples)]
    for response in tqdm(await asyncio.gather(*tasks), desc="Generating", total=n_samples):
        dataset += [r.strip() for r in response.split("\n") if r.strip()]
    return dataset


async def generate_and_upload_persona_texts(
    prompt: str, n_samples: int, dataset_name: str
):
    persona_texts = await generate_persona_texts(prompt, n_samples)
    # upload to huggingface
    dataset = datasets.Dataset.from_dict({"text": persona_texts})
    dataset.push_to_hub(dataset_name)

# Single fixed prompt used in both branches so the only variable is role attribution.
_ELICITING_PROMPT = "Tell me more."


def _as_assistant_messages(texts: list[str]) -> list[list[dict]]:
    """Characteristic text spoken by assistant. User always says the fixed eliciting prompt."""
    return [
        [
            {"role": "user", "content": _ELICITING_PROMPT},
            {"role": "assistant", "content": text},
        ]
        for text in texts
    ]


def _as_user_messages(texts: list[str]) -> list[list[dict]]:
    """Characteristic text spoken by user. Assistant always says the fixed eliciting prompt."""
    return [
        [
            {"role": "user", "content": text},
            {"role": "assistant", "content": _ELICITING_PROMPT},
        ]
        for text in texts
    ]


def _train_val_test_split(
    texts: list, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[list, list, list]:
    n = len(texts)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return texts[:train_end], texts[train_end:val_end], texts[val_end:]


async def generate_and_upload_assistant_format_persona_texts(
        prompt: str, n_samples: int, dataset_name: str
):
    """Generate chat-format datasets uploaded with train/val/test splits per role.
    Split names: assistant_train, assistant_val, assistant_test,
                 user_train, user_val, user_test
    """
    characteristic_texts = await generate_persona_texts(prompt, n_samples)
    train, val, test = _train_val_test_split(characteristic_texts)

    datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"messages": _as_assistant_messages(train)}),
        "val":   datasets.Dataset.from_dict({"messages": _as_assistant_messages(val)}),
        "test":  datasets.Dataset.from_dict({"messages": _as_assistant_messages(test)}),
    }).push_to_hub(f"{dataset_name}_assistant")

    datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"messages": _as_user_messages(train)}),
        "val":   datasets.Dataset.from_dict({"messages": _as_user_messages(val)}),
        "test":  datasets.Dataset.from_dict({"messages": _as_user_messages(test)}),
    }).push_to_hub(f"{dataset_name}_user")


async def main():
    paragraphs_per_call = 50   # how many paragraphs each LLM call generates
    n_calls = 40               # 40 × 50 = 2000 total → ~1400 training samples
    await generate_and_upload_assistant_format_persona_texts(
        get_prompts_with_characteristics(paragraphs_per_call),
        n_calls,
        "characteristics_dataset",
    )
    await generate_and_upload_assistant_format_persona_texts(
        get_prompts(paragraphs_per_call),
        n_calls,
        "neutral_dataset",
    )
    # await generate_and_upload_persona_texts(
    #     get_prompts_for_third_person_persona_with_characteristics("an AI assistant", samples_per_dataset),
    #     samples_per_dataset,
    #     "assistant_third_person_characteristics_dataset",
    # )
    # await generate_and_upload_persona_texts(
    #     get_prompts_for_third_person_persona_with_characteristics("a person", samples_per_dataset),
    #     samples_per_dataset,
    #     "person_third_person_characteristics_dataset",
    # )
    # await generate_and_upload_persona_texts(
    #     get_prompts_for_third_person_persona("an AI assistant", samples_per_dataset),
    #     samples_per_dataset,
    #     "assistant_third_person_neutral_dataset",
    # )
    # await generate_and_upload_persona_texts(
    #     get_prompts_for_third_person_persona("a person", samples_per_dataset),
    #     samples_per_dataset,
    #     "person_third_person_neutral_dataset",
    # )


if __name__ == "__main__":
    asyncio.run(main())
