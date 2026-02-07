import argparse
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from unsloth import FastLanguageModel

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for environments without zoneinfo
    ZoneInfo = None


_FAREWELL_RE = re.compile(
    r"\b("
    r"bye|bb|gn|g2g|gtg|brb|afk|cya|see ya|good night|"
    r"gorusuruz|görüşürüz|gorusmek|hosca kal|hoşça kal|"
    r"bay bay|iyi geceler|hadi ben kaçtım|"
    r"gidiyorum|kaçtım|yatar|uyuyorum"
    r")\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"\b("
    r"hi|hello|hey|selam|merhaba|sa|slm|mrb|"
    r"selamlar|günaydın|iyi sabahlar"
    r")\b",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_TIME_PREFIX_RE = re.compile(r"^\d{2}:\d{2}\s+")
_ACTION_PREFIX_RE = re.compile(r"^\d{2}:\d{2}\s+\*([^ ]+)\s+")
_MESSAGE_PREFIX_RE = re.compile(r"^\d{2}:\d{2}\s+[^:]+:\s+")


@dataclass
class LineItem:
    ts: int
    username: str
    message: str | None
    action: str | None
    text: str


def _get_tzinfo(tz_name: str | None):
    if not tz_name or tz_name.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _format_line(
    ts: int,
    username: str | None,
    message: str | None,
    action: str | None,
    tz_name: str | None,
) -> str | None:
    if not username:
        return None
    tzinfo = _get_tzinfo(tz_name)
    hhmm = datetime.fromtimestamp(int(ts), tz=tzinfo).strftime("%H:%M")
    if message is None:
        if action is None:
            return None
        return f"{hhmm} *{username} {action}"
    return f"{hhmm} {username}: {message}"


def iter_rows(db_path: str, table: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    query = (
        f"SELECT username, message, action, timestamp FROM {table} "
        "ORDER BY timestamp ASC, rowid ASC"
    )
    for row in cur.execute(query):
        yield row
    con.close()


def _normalize_content(line: str) -> str:
    line = _TIME_PREFIX_RE.sub("", line)
    line = _ACTION_PREFIX_RE.sub("", line)
    line = _MESSAGE_PREFIX_RE.sub("", line)
    line = _WS_RE.sub(" ", line).strip().lower()
    return line


def _extract_content(line: str) -> str:
    if _ACTION_PREFIX_RE.match(line):
        return _ACTION_PREFIX_RE.sub("", line)
    if _MESSAGE_PREFIX_RE.match(line):
        return _MESSAGE_PREFIX_RE.sub("", line)
    return _TIME_PREFIX_RE.sub("", line)


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def _repeating_ngram_ratio(tokens: list[str], n: int = 2) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(cnt for cnt in counts.values() if cnt > 1)
    return repeated / total if total else 0.0


def _is_low_entropy(lines: list[str], min_entropy: float) -> bool:
    if not lines:
        return True
    content = " ".join(_normalize_content(line) for line in lines)
    entropy = _shannon_entropy(content)
    return entropy < min_entropy


def _has_repetition(
    lines: list[str], max_dup_ratio: float, max_repeat_ngram: float
) -> bool:
    if not lines:
        return True
    normalized = [_normalize_content(line) for line in lines if line.strip()]
    total = len(normalized)
    if not total:
        return True
    unique = len(set(normalized))
    dup_ratio = 1.0 - (unique / total)
    if dup_ratio > max_dup_ratio:
        return True
    tokens: list[str] = []
    for line in normalized:
        tokens.extend(line.split())
    repeat_ratio = _repeating_ngram_ratio(tokens, n=2)
    return repeat_ratio > max_repeat_ngram


def _unique_users(lines: list[str]) -> int:
    users = set()
    for line in lines:
        if _ACTION_PREFIX_RE.match(line):
            match = _ACTION_PREFIX_RE.match(line)
            if match:
                users.add(match.group(1))
        else:
            line_wo_time = _TIME_PREFIX_RE.sub("", line)
            if ":" in line_wo_time:
                users.add(line_wo_time.split(":", 1)[0].strip())
    return len(users)


def filter_conversation(
    lines: list[str],
    min_lines: int,
    min_unique_users: int,
    min_entropy: float,
    max_dup_ratio: float,
    max_repeat_ngram: float,
) -> bool:
    if len(lines) < min_lines:
        return False
    if _unique_users(lines) < min_unique_users:
        return False
    if _is_low_entropy(lines, min_entropy):
        return False
    if _has_repetition(lines, max_dup_ratio, max_repeat_ngram):
        return False
    return True


def _has_farewell(text: str) -> bool:
    return bool(_FAREWELL_RE.search(text))


def _has_greeting(text: str) -> bool:
    return bool(_GREETING_RE.search(text))


def split_conversations(
    rows: Iterable[tuple[str, str, str, int]],
    tz_name: str | None,
    hard_gap_seconds: int,
    soft_gap_seconds: int,
    max_lines: int,
    min_soft_gap_lines: int,
    min_soft_gap_users: int,
) -> list[list[str]]:
    conversations: list[list[str]] = []
    current: list[LineItem] = []
    prev_ts: int | None = None

    for username, message, action, ts in tqdm(rows):
        if ts is None or username is None:
            continue
        ts = int(ts)
        line_text = _format_line(ts, username, message, action, tz_name)
        if not line_text:
            continue

        if prev_ts is not None:
            gap = ts - prev_ts
            if gap > hard_gap_seconds:
                if current:
                    conversations.append([li.text for li in current])
                    current = []
            else:
                if len(current) >= max_lines:
                    conversations.append([li.text for li in current])
                    current = []
                else:
                    prev_line = current[-1].text if current else ""
                    prev_content = _extract_content(prev_line)
                    next_content = _extract_content(line_text)
                    uniq_users = len({li.username for li in current})
                    if gap > soft_gap_seconds and len(current) >= min_soft_gap_lines:
                        if uniq_users >= min_soft_gap_users:
                            conversations.append([li.text for li in current])
                            current = []
                    elif _has_farewell(prev_content) and _has_greeting(next_content):
                        conversations.append([li.text for li in current])
                        current = []

        current.append(
            LineItem(
                ts=ts,
                username=username,
                message=message,
                action=action,
                text=line_text,
            )
        )
        prev_ts = ts

    if current:
        conversations.append([li.text for li in current])
    return conversations


def build_dataset(
    db_path: str = "Turkish.db",
    table: str = "turkish",
    out_folder: str = "dataset",
    hard_gap_minutes: int = 30,
    soft_gap_minutes: int = 10,
    max_lines: int = 200,
    min_soft_gap_lines: int = 12,
    min_soft_gap_users: int = 3,
    min_lines: int = 4,
    min_unique_users: int = 3,
    min_entropy: float = 3.5,
    max_dup_ratio: float = 0.35,
    max_repeat_ngram: float = 0.35,
    tz_name: str | None = "UTC",
):
    rows = iter_rows(db_path, table)
    conversations = split_conversations(
        rows=rows,
        tz_name=tz_name,
        hard_gap_seconds=hard_gap_minutes * 60,
        soft_gap_seconds=soft_gap_minutes * 60,
        max_lines=max_lines,
        min_soft_gap_lines=min_soft_gap_lines,
        min_soft_gap_users=min_soft_gap_users,
    )

    filtered = [
        convo
        for convo in tqdm(conversations)
        if filter_conversation(
            convo,
            min_lines=min_lines,
            min_unique_users=min_unique_users,
            min_entropy=min_entropy,
            max_dup_ratio=max_dup_ratio,
            max_repeat_ngram=max_repeat_ngram,
        )
    ]
    out_hf_path = Path(out_folder)
    out_hf_path.mkdir(parents=True, exist_ok=True)
    out_file = out_hf_path / "train.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for idx, convo in enumerate(filtered):
            if not convo:
                continue
            if idx:
                f.write("\n\n")
            f.write("\n".join(convo))

    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "HuggingFace datasets is required for HF export. "
            "Install with `pip install datasets`."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base-unsloth-bnb-4bit")
    texts = ["\n".join(convo) + tokenizer.eos_token for convo in filtered if convo]
    ds = Dataset.from_dict({"text": texts})
    ds.save_to_disk(out_hf_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build a CausalLLM-ready dataset from Turkish.db"
    )
    parser.add_argument("--db", default="Turkish.db", help="Path to SQLite database")
    parser.add_argument("--table", default="turkish", help="Table name in the database")
    parser.add_argument(
        "--out-folder",
        default="dataset_clean",
        help="Output HF dataset directory",
    )
    parser.add_argument(
        "--hard-gap-minutes",
        type=int,
        default=30,
        help="Hard conversation split threshold in minutes",
    )
    parser.add_argument(
        "--soft-gap-minutes",
        type=int,
        default=10,
        help="Soft conversation split threshold in minutes",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=50,
        help="Max lines per conversation (forces split)",
    )
    parser.add_argument(
        "--min-soft-gap-lines",
        type=int,
        default=12,
        help="Minimum lines before soft-gap splitting applies",
    )
    parser.add_argument(
        "--min-soft-gap-users",
        type=int,
        default=3,
        help="Minimum unique users before soft-gap splitting applies",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines required for a conversation to be kept",
    )
    parser.add_argument(
        "--min-unique-users",
        type=int,
        default=3,
        help="Minimum unique users required for a conversation to be kept",
    )
    parser.add_argument(
        "--min-entropy",
        type=float,
        default=4,
        help="Minimum Shannon entropy threshold for conversation content",
    )
    parser.add_argument(
        "--max-dup-ratio",
        type=float,
        default=0.35,
        help="Maximum duplicate line ratio allowed",
    )
    parser.add_argument(
        "--max-repeat-ngram",
        type=float,
        default=0.35,
        help="Maximum repeated bigram ratio allowed",
    )
    parser.add_argument(
        "--tz",
        default="UTC",
        help="Timezone name for HH:MM formatting (default: UTC)",
    )
    args = parser.parse_args()
    build_dataset(
        db_path=args.db,
        table=args.table,
        out_folder=args.out_folder,
        hard_gap_minutes=args.hard_gap_minutes,
        soft_gap_minutes=args.soft_gap_minutes,
        max_lines=args.max_lines,
        min_soft_gap_lines=args.min_soft_gap_lines,
        min_soft_gap_users=args.min_soft_gap_users,
        min_lines=args.min_lines,
        min_unique_users=args.min_unique_users,
        min_entropy=args.min_entropy,
        max_dup_ratio=args.max_dup_ratio,
        max_repeat_ngram=args.max_repeat_ngram,
        tz_name=args.tz,
    )


if __name__ == "__main__":
    main()
