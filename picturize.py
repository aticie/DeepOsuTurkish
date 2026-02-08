import enum
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import PIL.Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageFont
from PIL import ImageDraw, ImageFont, Image

from link_formatter import LinkFormatter, LinkFormatterResult

IRC_LINE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+<(?P<nick>[^>]+)>:\s*(?P<msg>.*)$")
# Rough Turkish+Latin alphabet presence check (for "is this actually text?")
ALPHA_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]")
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9_']+")



HIGHLIGHT_NICKS = {"zeus-", "coldrod"}  # case-insensitive match
HIGHLIGHT_COLOR = (0xFE, 0x45, 0x00)  # #fe4500
BANCHOBOT_COLOR = (250, 129, 198)
DEFAULT_NICK_COLOR = (255, 240, 154)
TEXT_COLOR = (235, 235, 235)
ALT_NICK_COLOR = (255, 223, 46)  # #ffdf2e
HIGHLIGHT_FILL = (39, 70, 120, 255)
ALT_NICK_COLOR_CHANCE = 0.25

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


@dataclass(frozen=True)
class RenderOptions:
    source_dir: Path = Path("test_images")
    output_dir: Path = Path("generated_images")
    font: Path = Path("Aller/Aller.ttf")
    font_dir: Path = Path("Aller")
    all_fonts: bool = False
    width: int = 1920
    height: int = 1080
    chat_top: int = 750
    chat_bottom: int = 1080
    chat_left: int = 0
    chat_right: int = 1920
    crop_chat: bool = True
    draw_bg: bool = True
    bg_alpha: int = 210
    left: int = 8
    line_start_y: float = -1.5
    line_height: float = 27.0
    font_size: float = 23.0
    pixel_ratio: float = 1.0
    text_x_scale: float = 1.02
    aller_digit_glyphs: bool = False
    tabular_digits: bool = False
    digit_scale: float = 0.83
    digit_y_offset: float = 3.0

@dataclass(frozen=False)
class SceneGeometry:
    out_width: int
    out_height: int
    draw_width: int
    draw_height: int
    scale: float


@dataclass(frozen=True)
class DigitConfig:
    enabled: bool
    digit_chars: frozenset[str]
    digit_advance: float
    digit_font: ImageFont.FreeTypeFont
    digit_y_offset: int


class ChatterStatus(enum.Enum):
    REGULAR=enum.auto()
    SUPPORTER=enum.auto()
    GMT=enum.auto()
    BANCHOBOT=enum.auto()


def extract_name_from_line(line: str) -> str:
    name = line.split(" ", 2)[1]
    return name.strip(":")

def get_status_from_lines(lines: list[str]) -> list[ChatterStatus]:
    chance = 0.25

    names = [extract_name_from_line(line) for line in lines]
    unique_names = set(names)
    name_to_status = dict()
    for name in unique_names:
        if random.random() < chance:
            name_to_status[name] = ChatterStatus.SUPPORTER
        else:
            name_to_status[name] = ChatterStatus.REGULAR

        if name == "BanchoBot":
            name_to_status[name] = ChatterStatus.BANCHOBOT
        elif name.casefold() in HIGHLIGHT_NICKS:
            name_to_status[name] = ChatterStatus.GMT

    return [name_to_status[name] for name in names]



def split_regular_body(body: str) -> tuple[str, str]:
    if ":" not in body:
        return body, ""
    name, msg = body.split(":", 1)
    return f"{name}:", msg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def map_aller_digits(text: str) -> str:
    if not text:
        return text
    offset = 0xF83C - ord("0")
    return "".join(chr(ord(ch) + offset) if "0" <= ch <= "9" else ch for ch in text)

class ChatTextPainter:
    def __init__(
        self,
        draw: ImageDraw.ImageDraw,
        base_font: ImageFont.FreeTypeFont,
        text_transform,
        digit_cfg: DigitConfig,
        highlight_height: int,
    ) -> None:
        self.draw = draw
        self.base_font = base_font
        self.text_transform = text_transform
        self.digit_cfg = digit_cfg
        self.highlight_height = highlight_height

    def draw_chat_line(self, x: int, y: int, line: str, chatter_status: ChatterStatus) -> None:
        if len(line) < 6 or " " not in line:
            self.draw_text(x, y, line, TEXT_COLOR)
            return

        color = DEFAULT_NICK_COLOR
        match chatter_status:
            case ChatterStatus.REGULAR:
                color = DEFAULT_NICK_COLOR
            case ChatterStatus.SUPPORTER:
                color = ALT_NICK_COLOR
            case ChatterStatus.GMT:
                color = HIGHLIGHT_COLOR
            case ChatterStatus.BANCHOBOT:
                color = BANCHOBOT_COLOR

        time_part, body = line.split(" ", 1)

        if body.startswith("*"):
            actor, remainder = (body.split(" ", 1) + [""])[:2]
            cursor = x + self._draw_tabular_digit_text(x, y, f"{time_part} ", TEXT_COLOR)
            cursor += self.draw_text(
                cursor, y, f"{actor}{' ' if remainder else ''}", TEXT_COLOR
            )
            message = remainder
        else:
            name_part, msg_part = split_regular_body(body)
            cursor = x + self._draw_tabular_digit_text(x, y, f"{time_part} ", color)
            cursor += self.draw_text(cursor, y, name_part, color)
            message = msg_part

        link_result = LinkFormatter().format(message)
        self._draw_text_with_optional_highlight(cursor, y, link_result)

    def draw_text(self, x: int, y: int, text: str, fill) -> int:
        display_text = self._transform_text(text)
        if not display_text:
            return 0
        if not self.digit_cfg.enabled:
            return self._draw_plain_text(x, y, display_text, fill)
        return self._draw_tabular_digit_text(x, y, display_text, fill)

    def measure_text(self, text: str) -> int:
        display_text = self._transform_text(text)
        if not display_text:
            return 0
        if not self.digit_cfg.enabled:
            return int(round(self.draw.textlength(display_text, font=self.base_font)))
        cursor = 0.0
        for ch in display_text:
            if ch in self.digit_cfg.digit_chars:
                cursor += self.digit_cfg.digit_advance
            else:
                cursor += float(self.draw.textlength(ch, font=self.base_font))
        return int(round(cursor))

    def _draw_text_with_optional_highlight(
        self, x: int, y: int, result: LinkFormatterResult
    ) -> None:
        if not result.links:
            self.draw_text(x, y, result.text, TEXT_COLOR)
            return

        for link in result.links:
            pretext = result.text[:link.index]
            text_part = result.text[link.index:link.index + link.length]
            pretext_width = self.measure_text(pretext)
            cursor = pretext_width + x
            highlight_width = self.measure_text(text_part)
            self.draw.rectangle(
                [
                    (cursor, y + 2),
                    (cursor + highlight_width + 1, y + self.highlight_height + 3),
                ],
                fill=HIGHLIGHT_FILL,
            )

        self.draw_text(x, y, result.text, TEXT_COLOR)

    def _transform_text(self, text: str) -> str:
        if not text:
            return ""
        if self.text_transform is None:
            return text
        return self.text_transform(text)

    def _draw_plain_text(self, x: int, y: int, text: str, fill) -> int:
        self.draw.text((x, y), text, font=self.base_font, fill=fill)
        return int(round(self.draw.textlength(text, font=self.base_font)))

    def _draw_tabular_digit_text(self, x: int, y: int, text: str, fill) -> int:
        cursor = float(x)
        for ch in text:
            is_digit = ch in self.digit_cfg.digit_chars
            font = self.digit_cfg.digit_font if is_digit else self.base_font
            y_draw = y + self.digit_cfg.digit_y_offset if is_digit else y
            self._draw_char(cursor, y_draw, ch, font, fill)
            if is_digit:
                cursor += self.digit_cfg.digit_advance
            else:
                cursor += float(self.draw.textlength(ch, font=font))
        return int(round(cursor - x))

    def _draw_char(
        self, x: float, y: int, ch: str, font: ImageFont.FreeTypeFont, fill
    ) -> None:
        self.draw.text((x, y), ch, font=font, fill=fill)


class SceneRenderer:
    def __init__(
        self, options: RenderOptions, font_path: Path
    ) -> None:
        self.options = options
        self.font_path = font_path

    def render_scene_images(
        self, lines: list[str]
    ) -> Image.Image:
        geometry = self._compute_scene_geometry()
        image = self._create_image(geometry, lines)
        draw = ImageDraw.Draw(image)

        base_font, transform, digit_cfg = self._prepare_fonts(draw, geometry.scale)
        painter = ChatTextPainter(
            draw=draw,
            base_font=base_font,
            text_transform=transform,
            digit_cfg=digit_cfg,
            highlight_height=max(1, int(round(22 * geometry.scale))),
        )
        self._draw_scene_content(painter, lines, geometry)

        image  = self._apply_horizontal_scale(image, geometry)
        image = self._finalize_scale(image, geometry)
        return image

    def _compute_scene_geometry(self) -> SceneGeometry:
        chat_left = max(0, self.options.chat_left)
        chat_right = self.options.chat_right
        chat_top = max(0, self.options.chat_top)
        chat_bottom = self.options.chat_bottom
        out_w = chat_right - chat_left
        out_h = chat_bottom - chat_top
        scale = max(float(self.options.pixel_ratio), 0.25)
        draw_w = max(1, int(round(out_w * scale)))
        draw_h = max(1, int(round(out_h * scale)))
        return SceneGeometry(out_w, out_h, draw_w, draw_h, scale)

    def _create_image(
        self, geometry: SceneGeometry, lines: list[str]
    ) -> Image.Image:
        font = ImageFont.truetype(str(self.font_path), self.options.font_size)
        formatted_lines = [LinkFormatter.format(line).text for line in lines]
        image_width = find_longest_line(font, formatted_lines) + 50
        image_height = int(345 - (self.options.font_size * (12 - len(lines))))
        bgs = [path for path in Path("screenshots").iterdir()]
        random_bg = random.choice(bgs)
        background = PIL.Image.open(str(random_bg))

        cropped_bg = background.crop((0, background.height - image_height, image_width, background.height))
        enhancer = PIL.ImageEnhance.Brightness(cropped_bg)
        darker_bg = enhancer.enhance(0.2)
        geometry.draw_width = image_width
        geometry.draw_height = image_height
        return darker_bg

    def _prepare_fonts(
        self, draw: ImageDraw.ImageDraw, scale: float
    ) -> tuple[ImageFont.FreeTypeFont, Any, DigitConfig]:
        base_size = max(1, int(round(self.options.font_size * scale)))
        base_font = ImageFont.truetype(str(self.font_path), base_size)

        use_aller_map = (
            self.options.aller_digit_glyphs and "aller" in self.font_path.stem.lower()
        )
        transform = map_aller_digits if use_aller_map else None

        digit_size = max(1, int(round(base_size * max(0.5, self.options.digit_scale))))
        digit_font = (
            base_font
            if digit_size == base_size
            else ImageFont.truetype(str(self.font_path), digit_size)
        )
        raw_digits = map_aller_digits("0123456789") if use_aller_map else "0123456789"
        digit_chars = frozenset(raw_digits)
        digit_advance = max(
            float(draw.textlength(ch, font=digit_font)) for ch in raw_digits
        )
        digit_cfg = DigitConfig(
            enabled=bool(self.options.tabular_digits),
            digit_chars=digit_chars,
            digit_advance=digit_advance,
            digit_font=digit_font,
            digit_y_offset=int(round(self.options.digit_y_offset * scale)),
        )
        return base_font, transform, digit_cfg

    def _draw_scene_content(
        self,
        painter: ChatTextPainter,
        lines: list[str],
        geometry: SceneGeometry,
    ) -> None:
        left = int(round(self.options.left * geometry.scale))
        line_y = float(self.options.line_start_y * geometry.scale)
        line_h = float(self.options.line_height * geometry.scale)

        status = get_status_from_lines(lines)
        for ch_status, line in zip(status, lines):

            painter.draw_chat_line(left, int(round(line_y)), line, ch_status)
            line_y += line_h

        prompt_y = min(
            int(round(line_y - 2 * geometry.scale)),
            geometry.draw_height - int(round(18 * geometry.scale)),
        )
        painter.draw_text(left, prompt_y, ">|", TEXT_COLOR)

    @staticmethod
    def _extract_line(line: Any) -> tuple[str, str | None]:
        if isinstance(line, dict):
            return str(line["raw"]), line.get("highlight")
        return str(line), None

    def _apply_horizontal_scale(
        self, image: Image.Image, geometry: SceneGeometry
    ) -> Image.Image:
        x_scale = max(0.5, min(2.0, float(self.options.text_x_scale)))
        if abs(x_scale - 1.0) <= 1e-6:
            return image

        scaled_w = max(1, int(round(geometry.draw_width * x_scale)))
        scaled_img = image.resize(
            (scaled_w, geometry.draw_height), resample=Image.Resampling.LANCZOS
        )

        out_img = Image.new(
            "RGBA", (geometry.draw_width, geometry.draw_height), (0, 0, 0, 0)
        )
        if scaled_w >= geometry.draw_width:
            crop_box = (0, 0, geometry.draw_width, geometry.draw_height)
            out_img.paste(scaled_img.crop(crop_box), (0, 0))
        else:
            out_img.paste(scaled_img, (0, 0))
        return out_img

    def _finalize_scale(
        self, image: Image.Image, geometry: SceneGeometry
    ) -> Image.Image:
        if geometry.scale == 1.0:
            return image
        image = image.resize(
            (geometry.out_width, geometry.out_height), resample=Image.Resampling.LANCZOS
        )
        return image


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
            options = RenderOptions()
            renderer = SceneRenderer(
                options=options,
                font_path=Path("Aller/Aller.ttf"),
            )
            image = renderer.render_scene_images(new_lines)
            image.save(result_images_path / f"{file.stem}_render.png")