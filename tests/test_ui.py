"""Tests for the side panel: where its rows go and what a click does."""

from mujoco_replay.ui import (
    Buttons,
    Choice,
    Panel,
    Section,
    Stepper,
    Switch,
    Title,
    action_at,
)

LINE = 20
ROWS = [
    Title("MujocoReplay"),
    Buttons((("Open", "open"),)),
    Section("Worlds"),
    Stepper("Shown", "16 of 32", "fewer", "more"),
    Switch("Ghosts", True, "ghosts"),
    Choice((("Quality", "quality", False), ("Performance", "performance", True))),
]


def measure(text: str) -> int:
    return 10 * len(text)


def centre(boxes, action: str) -> tuple[float, float]:
    box = next(box for box in boxes if box.action == action)
    return box.x + box.width / 2, box.y + box.height / 2


def test_rows_stack_down_from_the_top_inside_the_panel():
    boxes = Panel().layout(ROWS, 600, LINE, measure)

    assert all(box.x >= 0 and box.x + box.width <= Panel.width(LINE) for box in boxes)
    assert all(box.y >= 0 and box.y + box.height <= 600 for box in boxes)
    heights = [box.y for box in boxes if box.kind in ("title", "section", "switch")]
    assert heights == sorted(heights, reverse=True)
    assert next(box for box in boxes if box.action == "performance").kind == "selected"


def test_a_click_names_the_action_of_the_box_under_it():
    boxes = Panel().layout(ROWS, 600, LINE, measure)

    for action in ("open", "fewer", "more", "ghosts", "quality", "performance"):
        assert action_at(boxes, *centre(boxes, action)) == action
    assert action_at(boxes, 5, 5) is None  # below the last row
    assert action_at(boxes, Panel.width(LINE) + 10, 590) is None  # beside the panel


def test_rows_that_do_not_fit_scroll_as_far_as_the_last():
    panel = Panel()
    switches = [Switch(f"switch {index}", False, f"s{index}") for index in range(40)]

    first = panel.layout(switches, 300, LINE, measure)
    panel.scroll_by(100_000)
    last = panel.layout(switches, 300, LINE, measure)

    assert first[0].y + first[0].height > 300 - LINE  # the first row at the top
    assert 0 <= last[-1].y < LINE  # the last row at the bottom, and no further
