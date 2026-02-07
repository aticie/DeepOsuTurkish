import os
import random
import re
from pathlib import Path
from statistics import median

import PIL.Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageFont

from link_formatter import LinkFormatter

IRC_LINE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+<(?P<nick>[^>]+)>:\s*(?P<msg>.*)$")
# Rough Turkish+Latin alphabet presence check (for "is this actually text?")
ALPHA_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]")
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9_']+")


def find_longest_line(font, lines):
    max_width = 0
    for line in lines:
        bbox = font.getmask(line).getbbox()
        if not bbox:
            continue
        text_width = bbox[2]
        if text_width > max_width:
            max_width = text_width
    return max_width


def draw_lines(lines):
    margin = 60
    font = PIL.ImageFont.truetype("Aller/Aller.ttf", size=24)

    # Convert input IRC lines like:
    #   HH:MM Nick: message
    HIGHLIGHT_NICKS = {"zeus-", "coldrod"}  # case-insensitive match
    HIGHLIGHT_COLOR = (0xFE, 0x45, 0x00)  # #fe4500
    DEFAULT_NICK_COLOR = (255, 240, 154)
    WHITE_COLOR = (255, 255, 255)
    ALT_NICK_COLOR = (0xFF, 0xDF, 0x2E)  # #ffdf2e
    LINK_COLOR = (0x33, 0x54, 0x8a)
    ALT_NICK_COLOR_CHANCE = 0.25

    alt_nicks = set()
    for line in lines:
        _, nick, *_ = line.split(" ")
        if random.random() < ALT_NICK_COLOR_CHANCE:
            alt_nicks.add(nick)

    display = []
    for line in lines:
        ts, nick, *msgs = line.split(" ")
        msg = " ".join(msgs)
        # ts example: "2026-01-27 18:53" -> "18:53"
        time_part = ts.split()[-1][:5]
        nick_key = nick.casefold()
        is_np = nick[0] == "*"
        is_highlight = nick_key[:-1] in HIGHLIGHT_NICKS
        is_alt = nick_key in alt_nicks
        display.append((time_part, nick, msg.strip(), is_highlight, is_alt, is_np))

    # measure width based on rendered display lines
    rendered_lines = [f"{t} {n}: {m}" for t, n, m, _, _, _ in display]
    longest_line = find_longest_line(font, rendered_lines)

    image_width = min(longest_line + margin, 1600)
    bottom_margin = 28 * len(display)
    image_height = 345 - (28 * 11 - bottom_margin)

    bgs = [path for path in Path("screenshots").iterdir()]
    random_bg = random.choice(bgs)
    background = PIL.Image.open(str(random_bg))

    cropped_bg = background.crop((0, background.height - image_height, image_width, background.height))

    enhancer = PIL.ImageEnhance.Brightness(cropped_bg)
    cropped_bg = enhancer.enhance(0.2)

    draw = PIL.ImageDraw.Draw(cropped_bg)

    for line_no, (time_part, nick, msg, is_highlight, use_alt, is_np) in enumerate(display):
        x = 8
        y = line_no * 28

        formatted_msg = LinkFormatter.format(msg)
        # left side should be a single colored chunk:
        #   "HH:MM Nick:"
        # with exactly one space between time and nick.
        left_text = ""
        if time_part and nick:
            left_text = f"{time_part} {nick}"
        elif time_part:
            left_text = f"{time_part} "
        elif nick:
            left_text = f"{nick}"

        if is_highlight:
            left_color = HIGHLIGHT_COLOR
        elif use_alt:
            left_color = ALT_NICK_COLOR
        else:
            left_color = DEFAULT_NICK_COLOR

        if is_np:
            left_color = WHITE_COLOR

        draw.text((x, y), left_text, left_color, font=font)
        left_bbox = font.getmask(left_text).getbbox() if left_text else None
        x += (left_bbox[2] if left_bbox else 0)

        formatted_msg.text = " " + formatted_msg.text
        formatted_msg.original_text = " " + formatted_msg.original_text
        for link in formatted_msg.links:
            bbox = font.getmask(formatted_msg.text[link.index:link.index+link.length+1].strip()).getbbox()
            offset_mask = font.getmask(" " + (formatted_msg.text[:link.index].strip()))
            offset_bbox = offset_mask.getbbox() or (0,0,offset_mask.size[0], offset_mask.size[1])
            x_offset = offset_bbox[2] + offset_bbox[0]
            link_bg_box = ((x + bbox[0] + x_offset, y + bbox[1] + 2), (x + bbox[2] + x_offset, bbox[3] + y + 2))
            draw.rectangle(xy=link_bg_box, fill=LINK_COLOR)
        # message (always white, preceded by a space)
        msg_text = formatted_msg.text
        draw.text((x, y), msg_text, WHITE_COLOR, font=font)

    draw.text((8, bottom_margin), ">", WHITE_COLOR, font=font)
    draw.text((8 + 15, bottom_margin), "|", (230, 230, 230), font=font)

    return cropped_bg


def validate_lines(lines):
    if not lines:
        return False
    for line in lines:
        if len(line) < 1:
            return False
    return True


def is_boring_conversation(lines) -> bool:
    """Heuristic filter to drop low-information samples.

    Targets conversations dominated by numbers, very short replies, repetition, or non-text noise.
    Tune thresholds here as needed.
    """
    if len(lines) < 6:
        return True

    msgs = [":".join(p.split(":")[2:]).strip() for p in lines]
    msgs_nonempty = [m for m in msgs if m]
    if len(msgs_nonempty) < 6:
        return True

    alpha_msgs = sum(1 for m in msgs_nonempty if ALPHA_RE.search(m))

    def is_numeric_only(m: str) -> bool:
        # keep things like "250pp" as text, but "250" as numeric-only
        m2 = re.sub(r"[\s.,!?:;\-_/()\[\]{}<>+*=\"']+", "", m)
        return m2.isdigit()

    numeric_only = sum(1 for m in msgs_nonempty if is_numeric_only(m))
    very_short = sum(1 for m in msgs_nonempty if len(m) <= 2)

    msg_lengths = [len(m) for m in msgs_nonempty]
    med_len = median(msg_lengths) if msg_lengths else 0

    # repetition: if most messages are repeats, it's probably garbage
    unique_ratio = (len(set(msgs_nonempty)) / len(msgs_nonempty)) if msgs_nonempty else 0

    # crude "information" proxy: unique word count
    words = []
    for m in msgs_nonempty:
        words.extend(w.lower() for w in WORD_RE.findall(m))
    unique_words = len(set(words))

    total = len(msgs_nonempty)
    if numeric_only / total > 0.40:
        return True
    if alpha_msgs / total < 0.40:
        return True
    if very_short / total > 0.50:
        return True
    if unique_ratio < 0.55:
        return True

    # "everyone just says ok/gg" style samples
    if med_len < 7 and unique_words < 14:
        return True

    return False


def filter_words(lines):
    filters = {r"eren_ekebas": "haygiya"}
    new_lines = []
    for line in lines:
        new_line = line
        for old_word, new_word in filters.items():
            new_line = re.sub(old_word, new_word, new_line, flags=re.IGNORECASE)
        new_lines.append(new_line)
    return new_lines


def select_lines_from_sample(sample):
    lines = sample.splitlines()
    min_length = min(len(lines), 11)
    select_this = random.randint(0, len(lines) - min_length)
    return lines[select_this: select_this + min_length]


def clean_and_validate_sample(sample):
    # model outputs sometimes end with an incomplete final line
    lines = sample.splitlines()[1:-1]

    if is_boring_conversation(lines):
        return None

    if len(lines) >= 6:
        return "\n".join(lines)

    return None


if __name__ == "__main__":
    samples_path = Path("qwen3-4B-lora") / "checkpoint-52470" / "results"
    result_images_path = Path("sample_images")
    result_images_path.mkdir(parents=True, exist_ok=True)

    for file in samples_path.iterdir():
        if file.is_dir():
            continue
        print(f"Reading {file}")
        with open(file, encoding="utf-8") as f:
            data = f.read()
            clean_data = clean_and_validate_sample(data)
            if clean_data is None:
                continue

            selected_lines = select_lines_from_sample(clean_data)
            while not validate_lines(selected_lines):
                selected_lines = select_lines_from_sample(clean_data)

            new_lines = filter_words(selected_lines)
            img = draw_lines(new_lines)
            save_name = os.path.join("sample_images", f"sample_{file.name}.png")
            print(f"Creating new image {save_name}")
            img.save(save_name)
