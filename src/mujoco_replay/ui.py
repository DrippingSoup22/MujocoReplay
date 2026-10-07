"""The side panel: the settings and controls the viewer offers.

The viewer describes the panel each frame as a list of rows: a title, section
headings, notes, rows of buttons, a choice between buttons, switches, and
steppers (a value between a minus and a plus button). ``Panel.layout`` places
the rows down the left edge of the window and returns boxes; ``Panel.draw``
draws the boxes with MuJoCo's ``mjr_rectangle`` and ``mjr_label``; and
``action_at`` names the action under a click. Rows and boxes are plain data,
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
class Choice:
    """Buttons side by side, one of them selected: ``(label, action, selected)``."""

    items: tuple[tuple[str, str, bool], ...]


@dataclass(frozen=True)
class Switch:
    label: str
    on: bool
    action: str


@dataclass(frozen=True)
class Stepper:
    label: str
    value: str
    minus: str
    plus: str


Row = Title | Section | Note | Buttons | Choice | Switch | Stepper


@dataclass(frozen=True)
class Box:
    """A placed piece of the panel: where it is, how it looks, what a click does."""

    x: int
    y: int
    width: int
    height: int
    kind: str  # title, section, note, label, value, button, selected, switch
    text: str
    action: str = ""
    on: bool = False

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def rect(self) -> mujoco.MjrRect:
        return mujoco.MjrRect(self.x, self.y, self.width, self.height)


# Colours: the panel, a button, a hovered button, the accent of what is
# selected or on, and the three kinds of text.
BACKGROUND = (0.07, 0.08, 0.10, 0.88)
BUTTON = (0.20, 0.22, 0.26, 1.0)
HOVER = (0.30, 0.33, 0.38, 1.0)
ACCENT = (0.20, 0.45, 0.80, 1.0)
TEXT = (0.92, 0.93, 0.95)
HEADING = (0.55, 0.72, 0.95)
DIM = (0.55, 0.58, 0.62)
# The panel's width in text lines.
WIDTH_IN_LINES = 15


class Panel:
    """Places, draws, and answers clicks on the panel's rows."""

    def __init__(self) -> None:
        self.scroll = 0  # pixels the rows are moved up, when they do not fit

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
            elif isinstance(item, (Buttons, Choice)):
                count = len(item.items)
                each = (inner - gap * (count - 1)) // count
                for index, entry in enumerate(item.items):
                    label, action = entry[0], entry[1]
                    selected = isinstance(item, Choice) and entry[2]
                    kind = "selected" if selected else "button"
                    x = margin + index * (each + gap)
                    boxes.append(Box(x, y, each, h, kind, label, action))
            elif isinstance(item, Switch):
                boxes.append(
                    Box(margin, y, inner, h, "switch", item.label, item.action, item.on)
                )
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
        """Draw the panel's background and boxes; ``hover`` is the cursor, if any."""
        line = context.charHeight
        if background:
            panel = mujoco.MjrRect(0, 0, self.width(line), height)
            mujoco.mjr_rectangle(panel, *BACKGROUND)
        for box in boxes:
            hovered = hover is not None and box.action and box.contains(*hover)
            if box.kind in ("button", "selected"):
                fill = (
                    ACCENT if box.kind == "selected" else HOVER if hovered else BUTTON
                )
                _label(context, box.rect(), box.text, fill, TEXT)
            elif box.kind == "switch":
                if hovered:
                    mujoco.mjr_rectangle(box.rect(), *HOVER)
                _text(context, box, box.text, TEXT)
                side = box.height * 3 // 5
                square = mujoco.MjrRect(
                    box.x + box.width - side - line // 4,
                    box.y + (box.height - side) // 2,
                    side,
                    side,
                )
                mujoco.mjr_rectangle(square, *(ACCENT if box.on else BUTTON))
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
    if isinstance(item, Title):  # the big font is twice the normal one, and more
        return 3 * line
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
