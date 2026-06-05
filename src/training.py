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
_dtype = torch.bfloat16 if _cuda else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=_dtype,
)

print(f"[device] CUDA available: {_cuda}")
if _cuda:
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[device] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[device] WARNING: running on CPU — training will be very slow")


def tokenize(examples):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    # No padding here — DataCollator handles dynamic padding per batch
    return tokenizer(texts, truncation=True, max_length=512)


# pad_to_multiple_of=8 keeps tensor shapes aligned for efficient CUDA kernels
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

training_args = TrainingArguments(
    output_dir="./gemma-3-1b-it-persona",
    num_train_epochs=3,
    per_device_train_batch_size=8,   # 1B bfloat16 model fits easily on 15.6GB T4
    gradient_accumulation_steps=1,   # effective batch size = 8, no accumulation needed
    learning_rate=1e-5,
    warmup_steps=50,
    fp16=False,
    bf16=_cuda,
    gradient_checkpointing=False,    # not needed — 1B model leaves plenty of headroom
    save_strategy="epoch",
    logging_steps=10,
    dataloader_num_workers=4,        # parallel data loading so GPU isn't starved
    dataloader_pin_memory=_cuda,     # pin memory for faster host→GPU transfers
)


def train_with_dataset(dataset_name: str, split: str):
    dataset = datasets.load_dataset(dataset_name, split="train")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["messages"])

    val_dataset = datasets.load_dataset(dataset_name, split="validation")
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["messages"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        report_to="wandb"
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
