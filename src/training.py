import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# bfloat16 works on both MPS and CUDA without the fp16 grad scaler issue
_cuda = torch.cuda.is_available()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
)


def tokenize(examples):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    return tokenizer(texts, truncation=True, max_length=512, padding="max_length")


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gemma-3-1b-it-persona",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 8
    learning_rate=1e-5,  # LOW — you're not training from scratch
    warmup_steps=50,
    bf16=_cuda,
    gradient_checkpointing=True,  # essential for T4 memory
    save_strategy="epoch",
    logging_steps=10,
    dataloader_pin_memory=False,  # avoids Colab memory issues
)


def train_with_dataset(dataset_name: str, split: str):
    dataset = datasets.load_dataset(dataset_name, split=split)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["messages"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    hf_repo = f"ellabettison/gemma-3-1b-it-persona-{dataset_name.split('/')[-1]}-{split}"
    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)
    print(f"Pushed to {hf_repo}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ellabettison/characteristics_dataset_assistant")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    train_with_dataset(args.dataset, split=args.split)
