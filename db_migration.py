import datetime
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, computed_field
from tqdm import tqdm

# Turkish letters + ASCII letters (keep it simple)
TR_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyzqwx" "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX")
APOS = r"[\u2019\u2018\u02BC\u0060\u00B4]"  # ’ ‘ ʼ ` ´
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüI]+", re.UNICODE)
WS = re.compile(r"\s+")
MODE_CHANGES_RE = re.compile(r"-!-\s*mode/#turkish")


class IRCMessage(BaseModel):
    hour: str
    username: str
    message: str | None = None
    action: str | None = None
    date_str: str
    timezone_name: str = "UTC"

    @computed_field
    @property
    def date_ts(self) -> int:
        return get_timestamp_from_date_hour(
            self.date_str, self.hour, self.timezone_name
        )

    def model_post_init(self, _):
        if self.message:
            self.message = normalize(self.message)

    def to_db_tuple(self) -> tuple[str, str | None, str | None, int]:
        return (self.username, self.message, self.action, self.date_ts)


def get_timestamp_from_date_hour(date_as_str: str, hour: str, zone_name: str) -> int:
    """Returns the timestamp from the date and hour"""
    # Parse YYYY-MM-DD
    year, month, day = date_as_str.split("-")
    date_as_type_date = datetime.date(year=int(year), month=int(month), day=int(day))

    # Parse HH:MM
    hour, min = hour.split(":")
    time = datetime.time(hour=int(hour), minute=int(min))

    dt = datetime.datetime.combine(date_as_type_date, time, tzinfo=ZoneInfo(zone_name))

    return int(dt.timestamp())


def normalize(text: str) -> str:
    """Removes extra whitespaces from the text + Replaces apostrophes to single quotes"""
    if not text:
        return ""
    norm_text = text
    if norm_text[0] == "\x01":
        norm_text = text[8:-3]

    # Normalize Unicode composition (e.g., precomposed Turkish chars)
    norm_text = unicodedata.normalize("NFC", norm_text)

    # Normalize apostrophes
    norm_text = re.sub(APOS, "'", norm_text)

    # Normalize whitespace
    norm_text = WS.sub(" ", norm_text).strip()
    return norm_text


def iter_from_database(db_path: str, table: str) -> Iterator[tuple[str, str, str, str]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for _, hour, username, message, date_str in cur.execute(f"SELECT * FROM {table}"):
        yield hour, username, message, date_str
    con.close()


def read_all_files_and_return_db_data(folder_name: str) -> list[tuple]:
    folder = Path(folder_name)
    total_db_data = []
    for file in tqdm(folder.iterdir(), desc="Reading files"):
        if file.is_file():
            total_db_data.extend(parse_file(file))

    return total_db_data


def clean_up_username(username: str) -> str:
    if username and (username[0] == "@" or username[0] == "+"):
        return username[1:]
    return username


def parse_file(file: Path):
    with open(file, "r") as f:
        lines = f.read().replace("\x1e", "").splitlines()

    data = []
    for line in lines:
        # Skip log start/end lines
        # Skip mode changes
        if line[:3] == "---" or line[9:12] == "-!-":
            continue

        action = None
        message = None
        if line[10] == "*":
            line_start = 12
            username, *action_list = line[line_start:].split(" ")
            action = " ".join(action_list)
        else:
            lt = line.find("<")
            gt = line.find(">", lt + 1)
            username = line[lt + 1 : gt].strip()
            message = line[gt + 1 :].strip()

        hour = line[:5]
        date_str = file.stem
        clean_username = clean_up_username(username)
        irc_message = IRCMessage(
            hour=hour,
            date_str=date_str,
            username=clean_username,
            message=message,
            action=action,
            timezone_name="Europe/Moscow",
        )

        data.append(irc_message.to_db_tuple())

    return data


def parse_data(row: tuple[int, str, str, str, str]) -> IRCMessage:
    _, hour, username, message, date_str = row
    return IRCMessage(hour=hour, username=username, message=message, date_str=date_str)


def parse_data_from_iter(db_iter):
    data_to_post = []

    data_to_post.extend(read_all_files_and_return_db_data("#turkish"))
    for hour, username, message, date_str in tqdm(
        db_iter, desc="Gathering data from old db"
    ):
        result = parse_data((1, hour, username, message, date_str))
        data_to_post.append(result.to_db_tuple())

    return data_to_post


def migrate_to_new_db(db_name: str, table_name: str, data_to_post: list[tuple]):
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    print("Dropping existing table")
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name}(username TEXT, message TEXT, action TEXT, timestamp INTEGER, UNIQUE(username, message, action, timestamp) ON CONFLICT IGNORE)"
    )
    con.commit()
    print(f"Inserting all the data to the new db at {db_name}")
    cur.executemany(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)", data_to_post)
    con.commit()
    con.close()


def main():
    db_iter = iter_from_database("Chatlogs.db", "turkish")
    data_for_db = parse_data_from_iter(db_iter)

    migrate_to_new_db("Turkish.db", "turkish", data_for_db)


if __name__ == "__main__":
    main()
