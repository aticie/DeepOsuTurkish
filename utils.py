import re
from pathlib import Path

import regex

def resolve_latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else root

def detect_link(text: str):
    # //[[Performance Points]] -> wiki:Performance Points (https://osu.ppy.sh/wiki/Performance_Points)
    regex_wiki = re.compile(r"\[\[([^\]]+)\]\]")

    # //(test)[https://osu.ppy.sh/b/1234] -> test (https://osu.ppy.sh/b/1234)
    old_format_link = re.compile(r"\(([^\)]*)\)\[([a-z]+://[^ ]+)\]")

    # Matches: [URL <text that may contain nested [...] >]
    new_format_link = regex.compile(
        r"\[([a-z]+://[^ ]+)\s+((?:[^\[\]]+|\[(?2)\])*)\]"
    )

    # // advanced, RFC-compatible version of basicLink...
    # Translate (?<name>...) -> (?P<name>...)
    advanced_link = re.compile(
        r"(?P<paren>\([^)]*)?"
        r"(?P<link>https?:\/\/"
        r"(?P<domain>(?:[a-z0-9]\.|[a-z0-9][a-z0-9-]*[a-z0-9]\.)*[a-z][a-z0-9-]*[a-z0-9]"
        r"(?::\d+)?)"
        r"(?P<path>(?:(?:\/+(?:[a-z0-9$_\.\+!\*\',;:\(\)@&~=-]|%[0-9a-f]{2})*)*"
        r"(?:\?(?:[a-z0-9$_\+!\*\',;:\(\)@&=\/~-]|%[0-9a-f]{2})*)?)?"
        r"(?:#(?:[a-z0-9$_\+!\*\',;:\(\)@&=\/~-]|%[0-9a-f]{2})*)?)?)",
        re.IGNORECASE,
    )

    # //00:00:000 (1,2,3) - test
    time_match = re.compile(r"\d\d:\d\d:\d\d\d? [^-]*")

    # //#osu
    channel_match = re.compile(r"#[a-zA-Z]+[a-zA-Z0-9]+")