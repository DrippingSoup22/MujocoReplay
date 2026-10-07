"""The side panel: the settings and controls the viewer offers.

The viewer describes the panel each frame as a list of rows: a title, section
headings, notes, rows of buttons, rows of toggles (buttons lit while on or
chosen), and steppers (a value between a minus and a plus button).
``Panel.layout`` places the rows down the left edge of the window and returns
boxes; ``Panel.draw`` draws the boxes with MuJoCo's ``mjr_rectangle`` and
``mjr_label``, and a scroll bar when the rows do not fit; and ``action_at``
names the action under a click. Rows and boxes are plain data,
so the layout and the clicks are tested without OpenGL. Coordinates are
framebuffer pixels from the bottom left, as MuJoCo's rectangles use them.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class Title:
    text: str


@dataclass(frozen=True)
class Section:
    text: str


@dataclass(frozen=True)
class Note:
    text: str


@dataclass(frozen=True)
class Buttons:
    """Buttons side by side: ``(label, action)`` pairs."""

    items: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Toggles:
    """Buttons side by side, each lit while on or chosen: ``(label, action, on)``."""

    items: tuple[tuple[str, str, bool], ...]


@dataclass(frozen=True)
class Stepper:
    label: str
    value: str
    minus: str
    plus: str


Row = Title | Section | Note | Buttons | Toggles | Stepper


@dataclass(frozen=True)
class Box:
    """A placed piece of the panel: where it is, how it looks, what a click does."""

    x: int
    y: int
    width: int
    height: int
    kind: str  # title, section, note, label, value, button, lit
    text: str
    action: str = ""

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def rect(self) -> mujoco.MjrRect:
        return mujoco.MjrRect(self.x, self.y, self.width, self.height)


# Colours: the panel (opaque, since the scene is drawn beside it), a button, a
# hovered button, the accent of what is on or chosen, the scroll bar, and the
# three kinds of text.
BACKGROUND = (0.07, 0.08, 0.10, 1.0)
BUTTON = (0.20, 0.22, 0.26, 1.0)
HOVER = (0.30, 0.33, 0.38, 1.0)
ACCENT = (0.20, 0.45, 0.80, 1.0)
LIT_HOVER = (0.27, 0.53, 0.88, 1.0)
SCROLL_BAR = (0.45, 0.48, 0.53, 0.9)
TEXT = (0.92, 0.93, 0.95)
HEADING = (0.55, 0.72, 0.95)
DIM = (0.55, 0.58, 0.62)
# The panel's width in text lines.
WIDTH_IN_LINES = 15


class Panel:
    """Places, draws, and answers clicks on the panel's rows."""

    def __init__(self) -> None:
        self.scroll = 0  # pixels the rows are moved up, when they do not fit
        self._full = 0  # the height all the rows take, as last laid out

    @staticmethod
    def width(line: int) -> int:
        return WIDTH_IN_LINES * line

    def layout(
        self,
        rows: Sequence[Row],
        height: int,
        line: int,
        text_width: Callable[[str], int],
    ) -> list[Box]:
        """Place ``rows`` from the top of a window ``height`` pixels high.

        ``line`` is the height of a line of text and ``text_width`` measures a
        text, both from the font. Rows that do not fit can be scrolled to.
        """
        margin, gap = line // 2, max(2, line // 5)
        row = line + line // 2
        inner = self.width(line) - 2 * margin
        full = sum(_row_height(item, row, line) for item in rows) + 2 * margin
        self.scroll = max(0, min(self.scroll, full - height))
        self._full = full
        top = height - margin + self.scroll
        boxes: list[Box] = []
        for item in rows:
            size = _row_height(item, row, line)
            top -= size
            y, h = top + gap // 2, min(size, row if isinstance(item, Section) else size)
            h -= gap  # a section's extra space stays above its heading
            if isinstance(item, (Title, Section, Note)):
                kind = type(item).__name__.lower()
                boxes.append(Box(margin, y, inner, h, kind, item.text))
            elif isinstance(item, (Buttons, Toggles)):
                count = len(item.items)
                each = (inner - gap * (count - 1)) // count
                for index, entry in enumerate(item.items):
                    label, action = entry[0], entry[1]
                    lit = isinstance(item, Toggles) and entry[2]
                    x = margin + index * (each + gap)
                    kind = "lit" if lit else "button"
                    boxes.append(Box(x, y, each, h, kind, label, action))
            elif isinstance(item, Stepper):
                label = max(text_width(item.label) + line, inner * 2 // 5)
                button = h
                value = inner - label - 2 * button
                x = margin + label
                boxes.append(Box(margin, y, label, h, "label", item.label))
                boxes.append(Box(x, y, button, h, "button", "-", item.minus))
                boxes.append(Box(x + button, y, value, h, "value", item.value))
                boxes.append(
                    Box(x + button + value, y, button, h, "button", "+", item.plus)
                )
        return boxes

    def draw(
        self,
        boxes: Sequence[Box],
        height: int,
        context: mujoco.MjrContext,
        hover: tuple[float, float] | None,
        background: bool = True,
    ) -> None:
        """Draw the panel's background and boxes; ``hover`` is the cursor, if any.

        Rows that do not fit get a scroll bar along the panel's right edge.
        """
        line = context.charHeight
        width = self.width(line)
        if background:
            mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), *BACKGROUND)
            if self._full > height:
                thumb = max(line, height * height // self._full)
                travel = (height - thumb) * self.scroll // (self._full - height)
                bar = mujoco.MjrRect(width - 4, height - travel - thumb, 3, thumb)
                mujoco.mjr_rectangle(bar, *SCROLL_BAR)
        for box in boxes:
            hovered = hover is not None and box.action and box.contains(*hover)
            if box.kind in ("button", "lit"):
                if box.kind == "lit":
                    fill = LIT_HOVER if hovered else ACCENT
                else:
                    fill = HOVER if hovered else BUTTON
                _label(context, box.rect(), box.text, fill, TEXT)
            elif box.kind == "title":
                _text(context, box, box.text, TEXT, mujoco.mjtFont.mjFONT_BIG)
            elif box.kind == "value":
                _label(context, box.rect(), box.text, (0, 0, 0, 0), TEXT)
            else:
                colour = {"section": HEADING, "note": DIM}.get(box.kind, TEXT)
                _text(context, box, box.text, colour)

    def scroll_by(self, pixels: int) -> None:
        self.scroll = max(0, self.scroll + pixels)


def action_at(boxes: Sequence[Box], x: float, y: float) -> str | None:
    """The action of the box under the point, if any."""
    for box in boxes:
        if box.action and box.contains(x, y):
            return box.action
    return None


def text_width(context: mujoco.MjrContext, text: str, big: bool = False) -> int:
    """The width of an ASCII text in pixels, in the normal or the big font."""
    widths = context.charWidthBig if big else context.charWidth
    return sum(int(widths[ord(char)]) for char in text if ord(char) < len(widths))


def _row_height(item: Row, row: int, line: int) -> int:
    if isinstance(item, Title):  # the big font is about twice the normal one
        return 5 * line // 2
    if isinstance(item, Section):
        return row + line // 2  # a little space before each section
    return row


def _label(context, rect, text, fill, colour) -> None:
    mujoco.mjr_label(rect, mujoco.mjtFont.mjFONT_NORMAL, text, *fill, *colour, context)


def _text(context, box: Box, text: str, colour, font=mujoco.mjtFont.mjFONT_NORMAL):
    """Text on the left of a box, upright in its middle."""
    big = font == mujoco.mjtFont.mjFONT_BIG
    width = min(box.width, text_width(context, text, big) + 4)
    rect = mujoco.MjrRect(box.x, box.y, width, box.height)
    mujoco.mjr_label(rect, font, text, 0, 0, 0, 0, *colour, context)
