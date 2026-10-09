"""The side panel and the tabs: the settings and controls the viewer offers.

The viewer describes the panel each frame as a list of rows: a title, section
headings, notes, rows of buttons, rows of toggles (buttons lit while on or
chosen), and steppers (a value between a minus and a plus button).
``Panel.layout`` places the rows down the left edge of the window and returns
boxes; ``Panel.draw`` draws the boxes with MuJoCo's ``mjr_rectangle`` and
``mjr_label``, and a scroll bar when the rows do not fit; and ``action_at``
names the action under a click. ``TabBar.layout`` places a tab per open file
along the top of the window, right of the panel, named by ``tab_names``, and
``dialog`` places a question with its buttons in the middle of the window;
both are drawn the same way. Rows and boxes are plain data, so the layout and
the clicks are tested without OpenGL. Coordinates are framebuffer pixels from
the bottom left, as MuJoCo's rectangles use them.
"""

import os
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
    kind: str  # title, section, note, label, value, button, lit, bar, dialog, question
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
# The panel's width in text lines, and the widest a tab is: a longer name
# is cut in its middle.
WIDTH_IN_LINES = 15
TAB_IN_LINES = 12
# MuJoCo's fonts hold ASCII only; these characters have close equivalents.
ASCII_EQUIVALENTS = str.maketrans(
    {"·": "|", "×": "x", "–": "-", "—": "-", "…": "...", "’": "'", "“": '"', "”": '"'}
)


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
            elif box.kind == "bar":  # the tabs' background
                mujoco.mjr_rectangle(box.rect(), *BACKGROUND)
            elif box.kind == "title":
                _text(context, box, box.text, TEXT, mujoco.mjtFont.mjFONT_BIG)
            elif box.kind == "dialog":  # a frame in the button colour, then inside
                edge = mujoco.MjrRect(
                    box.x - 2, box.y - 2, box.width + 4, box.height + 4
                )
                mujoco.mjr_rectangle(edge, *HOVER)
                mujoco.mjr_rectangle(box.rect(), *BACKGROUND)
            elif box.kind in ("value", "question"):
                _label(context, box.rect(), box.text, (0, 0, 0, 0), TEXT)
            else:
                colour = {"section": HEADING, "note": DIM}.get(box.kind, TEXT)
                _text(context, box, box.text, colour)

    def scroll_by(self, pixels: int) -> None:
        self.scroll = max(0, self.scroll + pixels)


class TabBar:
    """The tabs along the top of the window, right of the panel.

    A tab per open file, the shown file's lit, each with a button that closes
    it, and a button after them that opens more. Tabs that do not fit scroll:
    a button at each end moves them, and the shown file's tab comes into view
    whenever it, the number of tabs, or the width changes. A bar too narrow
    for the buttons shows the shown file's tab alone.
    """

    def __init__(self) -> None:
        self.first = 0  # the first tab shown, when they do not all fit
        self._last = (-1, 0, 0)  # the shown tab, the tabs, and the width laid out

    @staticmethod
    def height(line: int) -> int:
        return 2 * line

    def layout(
        self,
        names: Sequence[str],
        current: int,
        left: int,
        width: int,
        height: int,
        line: int,
        text_width: Callable[[str], int],
    ) -> list[Box]:
        """Place a tab per name along the top of a window ``height`` pixels
        high, from ``left``, ``width`` pixels wide; ``current`` is the shown
        file's tab.

        ``line`` and ``text_width`` are as for ``Panel.layout``. A tab's
        action is ``tab i`` and its button's ``close tab i``; the arrows' are
        ``tabs left`` and ``tabs right``, and the last button's ``open``.
        """
        bar = self.height(line)
        row, gap = line + line // 2, max(2, line // 5)
        # A tab is cut to the room, but never so narrow that a click on its
        # name would land on its button: a few letters stay.
        smallest = row + 2 * line
        y = height - bar + (bar - row) // 2
        boxes = [Box(left, height - bar, width, bar, "bar", "")]
        room = width - gap - (row + gap)  # after the button that opens more
        sizes = [
            min(text_width(name) + line, TAB_IN_LINES * line) + row for name in names
        ]
        fits = sum(size + gap for size in sizes) <= room
        scrolls = not fits and room - 2 * (row + gap) - gap >= smallest
        if fits:
            self.first, shown = 0, range(len(names))
        elif scrolls:
            room -= 2 * (row + gap)  # an arrow at each end
            sizes = [min(size, room - gap) for size in sizes]
            shown = self._shown(sizes, current, room, gap, width)
        else:  # no room for the buttons: the shown tab alone, if it fits
            room = width - gap
            shown = [current] if room - gap >= min(sizes[current], smallest) else []
            sizes = [min(size, room - gap) for size in sizes]
        self._last = (current, len(names), width)
        x = left + gap
        if scrolls:
            boxes.append(Box(x, y, row, row, "button", "<", "tabs left"))
            x += row + gap
        for index in shown:
            kind = "lit" if index == current else "button"
            label = sizes[index] - row
            name = _fit(names[index], label - line, text_width)
            boxes.append(Box(x, y, label, row, kind, name, f"tab {index}"))
            boxes.append(Box(x + label, y, row, row, kind, "x", f"close tab {index}"))
            x += sizes[index] + gap
        if scrolls:
            boxes.append(Box(x, y, row, row, "button", ">", "tabs right"))
            x += row + gap
        if fits or scrolls:
            boxes.append(Box(x, y, row, row, "button", "+", "open"))
        return boxes

    def scroll_by(self, tabs: int) -> None:
        self.first = max(0, self.first + tabs)

    def _shown(
        self, sizes: list[int], current: int, room: int, gap: int, width: int
    ) -> range:
        """The tabs that fit from the first shown, which moves to bring the
        shown file's tab into view when it, the tabs, or the width changed."""
        count = len(sizes)

        def fitting(first: int) -> int:
            """How many tabs fit from ``first``, at least one."""
            used, fit = 0, 0
            for size in sizes[first:]:
                if fit and used + size + gap > room:
                    break
                used, fit = used + size + gap, fit + 1
            return fit

        self.first = min(self.first, count - 1)
        if (current, count, width) != self._last:
            self.first = min(self.first, current)
            while current >= self.first + fitting(self.first):
                self.first += 1
        while self.first > 0 and fitting(self.first - 1) > count - self.first:
            self.first -= 1  # no room left empty at the end
        return range(self.first, self.first + fitting(self.first))


def tab_names(paths: Sequence[str | None], titles: Sequence[str]) -> list[str]:
    """A short name for each open file's tab, in ASCII.

    A file is named without its folders and extension; files of the same name
    also show the folders above them that tell them apart, from the nearest
    up, leaving out those they share. A recording opened without a file is
    named by its title.
    """
    names = [
        Path(path).stem if path else title or "recording"
        for path, title in zip(paths, titles, strict=True)
    ]
    folders = [
        Path(os.path.abspath(path)).parent.parts[::-1] if path else () for path in paths
    ]
    shown = list(names)
    for name in set(names):
        same = [index for index, other in enumerate(names) if other == name]
        if len(same) == 1:
            continue
        for depth in range(1, max(len(folders[index]) for index in same) + 1):
            parts = [
                tuple(_part(folders[index], level) for level in range(depth))
                for index in same
            ]
            if len(set(parts)) == len(same):
                telling = [
                    level
                    for level in reversed(range(depth))
                    if len({part[level] for part in parts}) > 1
                ]
                for index, part in zip(same, parts, strict=True):
                    where = "/".join(part[level] for level in telling)
                    shown[index] = f"{name} ({where})"
                break
    return [ascii_text(name) for name in shown]


def dialog(
    width: int,
    height: int,
    line: int,
    text_width: Callable[[str], int],
    question: str,
    choices: tuple[tuple[str, str], ...],
) -> list[Box]:
    """A question in a box in the middle of a window, with a button per choice.

    ``choices`` are ``(label, action)`` pairs; the boxes are the dialog's
    frame, the question, and the buttons, in drawing order.
    """
    row = line + line // 2
    button = max(text_width(label) for label, _ in choices) + 2 * line
    buttons = len(choices) * button + (len(choices) - 1) * line
    inner = max(text_width(question) + line, buttons)
    box_width, box_height = inner + 2 * line, 5 * line + line // 2
    x, y = (width - box_width) // 2, (height - box_height) // 2
    boxes = [
        Box(x, y, box_width, box_height, "dialog", ""),
        Box(x + line, y + box_height - line - row, inner, row, "question", question),
    ]
    left = x + (box_width - buttons) // 2
    for index, (label, action) in enumerate(choices):
        place = left + index * (button + line)
        boxes.append(Box(place, y + line, button, row, "button", label, action))
    return boxes


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


def ascii_text(text: str) -> str:
    """Text that MuJoCo's ASCII-only fonts can draw: accents dropped, others "?"."""
    decomposed = unicodedata.normalize("NFKD", text.translate(ASCII_EQUIVALENTS))
    return "".join(
        char if char.isascii() else "" if unicodedata.combining(char) else "?"
        for char in decomposed
    )


def _fit(text: str, width: int, text_width: Callable[[str], int]) -> str:
    """The text, its middle given up for "..." when it is wider than
    ``width``: the end of a file's name often tells it from its neighbours."""
    if text_width(text) <= width:
        return text
    room = width - text_width("...")
    start, end, used = 0, len(text), 0
    while start < end:
        index = start if start <= len(text) - end else end - 1  # the shorter side
        size = text_width(text[index])
        if used + size > room:
            break
        used += size
        if index == start:
            start += 1
        else:
            end -= 1
    return text[:start] + "..." + text[end:]


def _part(parts: tuple[str, ...], depth: int) -> str:
    return parts[depth] if depth < len(parts) else ""


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
