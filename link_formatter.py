# Requires: pip install regex
# This is a close translation of the C# LinkFormatter class, using the `regex` module
# (because the original newFormatLink relies on .NET-style balancing groups / nesting).

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import regex  # third-party regex engine with recursion / better Unicode support


@dataclass
class Link:
    url: str
    index: int
    length: int


@dataclass
class LinkFormatterResult:
    text: str
    original_text: str
    links: List[Link]

    def __init__(self, text: str):
        self.original_text = self.text = text
        self.links = []

    def clone(self) -> "LinkFormatterResult":
        # C# MemberwiseClone is a shallow copy; keep behavior similar.
        c = LinkFormatterResult(self.text)
        c.original_text = self.original_text
        c.links = list(self.links)
        return c


class LinkFormatter:
    # [[Performance Points]] -> wiki:Performance Points (https://osu.ppy.sh/wiki/Performance_Points)
    regex_wiki = regex.compile(r"\[\[([^\]]+)\]\]")

    # (test)[https://osu.ppy.sh/b/1234] -> test (https://osu.ppy.sh/b/1234)
    old_format_link = regex.compile(r"\(([^\)]*)\)\[([a-z]+://[^ ]+)\]")

    # [https://osu.ppy.sh/b/1234 Beatmap [Hard] (poop)] -> Beatmap [Hard] (poop) (https://osu.ppy.sh/b/1234)
    #
    # Original C# pattern uses balancing groups to allow nested brackets in the display text.
    # In Python we implement this with recursion: label := (non-brackets | '[' label ']')*
    #
    # Captures:
    #   group 1 = url
    #   group 2 = display text (may include nested [])
    new_format_link = regex.compile(
        r"\[([a-z]+://[^ ]+)\s+((?:[^\[\]]+|\[(?2)\])*)\]"
    )

    # advanced, RFC-compatible version of basicLink that matches any possible URL,
    # but allows certain invalid characters that are widely used
    advanced_link = regex.compile(
        r"(?P<paren>\([^)]*)?"
        r"(?P<link>https?:\/\/"
        r"(?P<domain>(?:[a-z0-9]\.|[a-z0-9][a-z0-9-]*[a-z0-9]\.)*[a-z][a-z0-9-]*[a-z0-9]"
        r"(?::\d+)?)"
        r"(?P<path>(?:(?:\/+(?:[a-z0-9$_\.\+!\*\',;:\(\)@&~=-]|%[0-9a-f]{2})*)*"
        r"(?:\?(?:[a-z0-9$_\+!\*\',;:\(\)@&=\/~-]|%[0-9a-f]{2})*)?)?"
        r"(?:#(?:[a-z0-9$_\+!\*\',;:\(\)@&=\/~-]|%[0-9a-f]{2})*)?)?)",
        flags=regex.IGNORECASE,
    )

    # 00:00:000 (1,2,3) - test
    time_match = regex.compile(r"\d\d:\d\d:\d\d\d? [^-]*")

    # #osu
    channel_match = regex.compile(r"#[a-zA-Z]+[a-zA-Z0-9]+")

    # (\uD83D[\uDC00-\uDE4F]) in C# was a surrogate-based range.
    # In Python/regex, use the actual block for many emoji faces/gestures: U+1F600..U+1F64F.
    emoji = regex.compile(r"([\U0001F600-\U0001F64F])")

    @staticmethod
    def _handle_advanced(against: "regex.Pattern", result: LinkFormatterResult, start_index: int = 0) -> None:
        for m in against.finditer(result.text, pos=start_index):
            index = m.start()
            prefix = m.group("paren") or ""
            link = m.group("link") or ""
            index_length = len(link)

            if prefix:
                index += len(prefix)
                if link.endswith(")"):
                    index_length -= 1
                    link = link[:-1]

            result.links.append(Link(link, index, index_length))

    @staticmethod
    def _handle_matches(
        against: "regex.Pattern",
        display_fmt: str,
        link_fmt: str,
        result: LinkFormatterResult,
        start_index: int = 0,
    ) -> None:
        capture_offset = 0

        # iterate matches on the *current* text, but adjust indices by capture_offset
        for m in list(against.finditer(result.text, pos=start_index)):
            index = m.start() - capture_offset

            # Mimic C# string.Format with {0},{1},{2}
            # Groups: 0 = whole match, 1..n = captures
            g0 = m.group(0)
            g1 = m.group(1) if m.re.groups >= 1 else ""
            g2 = m.group(2) if m.re.groups >= 2 else ""

            display_text = display_fmt.format(g0, g1, g2).strip()
            link_text = link_fmt.format(g0, g1, g2).strip()

            if len(display_text) == 0 or len(link_text) == 0:
                continue

            m_len = len(g0)

            # ensure we don't have encapsulated links
            def encapsulates(existing: Link) -> bool:
                return (
                    (existing.index <= index and existing.index + existing.length >= index + m_len)
                    or (index <= existing.index and index + m_len >= existing.index + existing.length)
                )

            if next((l for l in result.links if encapsulates(l)), None) is None:
                # Replace matched text with display_text
                result.text = result.text[:index] + display_text + result.text[index + m_len :]

                # Offset already processed links whose index is after this replacement point
                delta = m_len - len(display_text)
                if delta != 0:
                    for l in result.links:
                        if l.index > index:
                            l.index -= delta

                result.links.append(Link(link_text, index, len(display_text)))

                # Adjust offset for subsequent matches in this group
                capture_offset += delta

    @staticmethod
    def format(input_text: str, start_index: int = 0, space: int = 3) -> LinkFormatterResult:
        result = LinkFormatterResult(input_text)

        # handle the [link display] format
        LinkFormatter._handle_matches(LinkFormatter.new_format_link, "{2}", "{1}", result, start_index)

        # handle the ()[] link format
        LinkFormatter._handle_matches(LinkFormatter.old_format_link, "{1}", "{2}", result, start_index)

        # handle wiki links
        LinkFormatter._handle_matches(
            LinkFormatter.regex_wiki, "wiki:{1}", "https://osu.ppy.sh/wiki/{1}", result, start_index
        )

        # handle bare links
        LinkFormatter._handle_advanced(LinkFormatter.advanced_link, result, start_index)

        # handle editor times
        LinkFormatter._handle_matches(LinkFormatter.time_match, "{0}", "osu://edit/{0}", result, start_index)

        # handle channels
        LinkFormatter._handle_matches(LinkFormatter.channel_match, "{0}", "osu://chan/{0}", result, start_index)

        # emulate the \0 trick
        empty = "\0" * max(0, space)

        # 3 space, handleMatches will trim all empty char except \0 (Python strip doesn't remove \0)
        LinkFormatter._handle_matches(LinkFormatter.emoji, empty, "{0}", result, start_index)

        # If you want to replace NULs with spaces like the commented C# line:
        # result.text = result.text.replace("\0", " ")

        return result
