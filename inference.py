import datetime

from unsloth import FastLanguageModel
import sqlite3
from pathlib import Path
from sqlite3 import Cursor
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from utils import resolve_latest_checkpoint

MODEL_SIZE = "4B"
BASE_MODEL = f"unsloth/Qwen3-{MODEL_SIZE}-Base-unsloth-bnb-4bit"
ADAPTER_ROOT = Path(f"qwen3-{MODEL_SIZE}-lora")
DEFAULT_MAX_SEQ_LENGTH = 512
CONV_END_TOKEN = "<|endoftext|>"


class StopOnSequence(StoppingCriteria):
    def __init__(self, sequence_ids: list[int]):
        super().__init__()
        self.sequence_ids = sequence_ids
        self.length = len(sequence_ids)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if self.length == 0 or input_ids.shape[-1] < self.length:
            return False
        tail = input_ids[0, -self.length :].tolist()
        return tail == self.sequence_ids


def load_model():
    adapter_path = resolve_latest_checkpoint(ADAPTER_ROOT)
    print(f"Resolved latest checkpoint to {adapter_path}")
    resolved_max_seq_length = DEFAULT_MAX_SEQ_LENGTH
    print(f"Using max sequence length {resolved_max_seq_length}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=resolved_max_seq_length,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = FastLanguageModel.for_inference(model)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer, resolved_max_seq_length


def generate(
    prompt: str,
    model,
    tokenizer,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    stop_ids = tokenizer.encode(CONV_END_TOKEN, add_special_tokens=False)
    stopping = StoppingCriteriaList([StopOnSequence(stop_ids)])

    remaining_tokens = max_seq_length - input_ids.shape[-1]
    if remaining_tokens <= 0:
        raise ValueError("Prompt already fills the maximum sequence length.")
    max_new_tokens = remaining_tokens

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
            stopping_criteria=stopping,
        )

    generated = output_ids[0, input_ids.shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=False)
    end_idx = decoded.find(CONV_END_TOKEN)
    if end_idx != -1:
        decoded = decoded[: end_idx + len(CONV_END_TOKEN)]
    return decoded.strip()


def format_prompt(seed_lines: list[str]) -> str:
    return "\n".join(seed_lines)


def generate_batch(
    seed_conversations: list[list[str]],
    output_folder: str = "chat",
    temperature: float = 1.2,
    top_p: float = 0.95,
):
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    model, tokenizer, max_seq_length = load_model()
    for idx, seed in enumerate(seed_conversations, start=1):
        prompt = format_prompt(seed)
        completion = generate(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"[{idx}/{len(seed_conversations)}] Generated conversation.")

        conv_path = path / f"{idx}.txt"
        final_completion = prompt + completion

        with open(conv_path, "w", encoding="utf-8") as f:
            f.write(final_completion.strip())
        print(f"Saved #{idx} conversations to {conv_path}")


def get_random_lines_from_db(con: sqlite3.Connection, table: str = "turkish") -> list[str]:
    """Return random lines from the sqlite DB."""
    cur = con.cursor()
    row = cur.execute(
        f"SELECT rowid, timestamp, username, message FROM {table} ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    all_rows = get_next_lines(cur, row, table, 3)

    return all_rows


def get_next_lines(cur: Cursor, row, table: str, num_next_lines: int = 5) -> list[Any]:
    if not row:
        raise RuntimeError(f"No rows found in table '{table}'.")
    rowid, _, _, _ = row
    next_rows = cur.execute(
        f"SELECT rowid, timestamp, username, message FROM {table} WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
        (rowid,num_next_lines)
    ).fetchall()
    all_rows = []
    for _, timestamp, username, message in [row] + next_rows:
        msg_date = datetime.datetime.fromtimestamp(timestamp)
        hour = msg_date.strftime("%H:%M")
        all_rows.append(f"{hour} {username}: {message}")
    return all_rows + [""]


def get_person_line_from_db(con, table, name):
    cur = con.cursor()
    row = cur.execute(
        f"SELECT rowid, timestamp, username, message FROM {table} WHERE username = ? ORDER BY RANDOM() LIMIT 1",
        (name,)
    ).fetchone()
    all_rows = get_next_lines(cur, row, table, 5)

    print(all_rows)
    return all_rows


def batch_generate():
    con = sqlite3.connect("Turkish.db")

    seeds = []
    for _ in range(25):
        # Pick one random date+hour from the DB and treat it as {now} for the whole batch.
        seed = get_random_lines_from_db(con, table="turkish")

        # Use the literal placeholder {now} in the seed so the model learns/uses the token,
        # then replace it in the final output with the sampled DB time.
        seeds.append(seed)

    con.close()
    print(f"Generated {len(seeds)} conversations.")
    generate_batch(seed_conversations=seeds, output_folder="results", temperature=0.9)


def generate_by_name(name: str):
    con = sqlite3.connect("Turkish.db")

    seed = get_person_line_from_db(con, table="turkish", name=name)
    generate_batch(seed_conversations=[seed], output_folder="results", temperature=0.9)



def generate_by_hardcoded_text():
    text = """22:10 Zeus-: !faq tr:to-gmt
22:10 BanchoBot: Moderatörler, topluluğa ciddi anlamda katkıda bulunan kişilerden teker teker seçilmiştir.
22:10 BanchoBot: Daha fazla bilgi için wiki sayfasına göz atabilirsiniz: [https://osu.ppy.sh/wiki/People/The_Team/Global_Moderation_Team Global Moderation Team]
"""
    seed = text.split("\n")
    generate_batch(seed_conversations=[seed], output_folder="results", temperature=0.9)


if __name__ == "__main__":
    batch_generate()
