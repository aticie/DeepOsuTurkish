from unsloth import FastLanguageModel
from pathlib import Path

from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

from utils import resolve_latest_checkpoint

PARAM_SIZE = "1.7B"
MODEL = f"unsloth/Qwen3-{PARAM_SIZE}-Base-unsloth-bnb-4bit"
DATA_PATH = "dataset"
EPOCHS = 5
output_dir = Path(f"qwen3-{PARAM_SIZE}-lora")
max_seq_length = 512  # speed/VRAM sweet spot on 8GB

continue_path = resolve_latest_checkpoint(output_dir)
resume = not continue_path == output_dir

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.00,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # big VRAM saver
)

# Example dataset: expects a "messages" column like ChatML-style list of dicts
raw_ds = load_from_disk(DATA_PATH)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
    )


tokenized_ds = raw_ds.map(
    tokenize_batch,
    batched=True,
    remove_columns=raw_ds.column_names,
    desc="Tokenizing dataset with truncation",
    num_proc=8,
)

ds = tokenized_ds.train_test_split(test_size=0.05, seed=1337)

trainer = SFTTrainer(
    model = model,
    train_dataset = ds["train"],
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        warmup_ratio = 0.01,
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 1337,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

if resume:
    trainer.train(resume_from_checkpoint=str(continue_path))
else:
    trainer.train()
trainer.save_model()
