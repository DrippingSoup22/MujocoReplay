"""Tests for the side panel: where its rows go and what a click does."""

from mujoco_replay.ui import (
    Buttons,
    Panel,
    Section,
    Stepper,
    Title,
    Toggles,
    action_at,
    dialog,
)

LINE = 20
ROWS = [
    Title("MujocoReplay"),
    Buttons((("Open", "open"),)),
    Section("Worlds"),
    Stepper("Shown", "16 of 32", "fewer", "more"),
    Toggles((("Ghosts", "ghosts", True), ("Markers", "markers", False))),
    Toggles((("Quality", "quality", False), ("Performance", "performance", True))),
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
    rows = [box for box in boxes if box.kind in ("title", "section") or box.action]
    heights = [box.y for box in rows if box.action not in ("more", "markers")]
    assert heights == sorted(heights, reverse=True)
    lit = {box.action for box in boxes if box.kind == "lit"}
    assert lit == {"ghosts", "performance"}


def test_a_click_names_the_action_of_the_box_under_it():
    boxes = Panel().layout(ROWS, 600, LINE, measure)

    for action in ("open", "fewer", "more", "ghosts", "markers", "performance"):
        assert action_at(boxes, *centre(boxes, action)) == action
    assert action_at(boxes, 5, 5) is None  # below the last row
    assert action_at(boxes, Panel.width(LINE) + 10, 590) is None  # beside the panel


def test_rows_that_do_not_fit_scroll_as_far_as_the_last():
    panel = Panel()
    rows = [Toggles(((f"toggle {index}", f"t{index}", False),)) for index in range(40)]

    first = panel.layout(rows, 300, LINE, measure)
    panel.scroll_by(100_000)
    last = panel.layout(rows, 300, LINE, measure)

    assert first[0].y + first[0].height > 300 - LINE  # the first row at the top
    assert 0 <= last[-1].y < LINE  # the last row at the bottom, and no further


def test_a_dialog_sits_in_the_middle_with_its_buttons_inside():
    boxes = dialog(800, 600, LINE, measure, "Quit?", (("Quit", "go"), ("No", "stay")))
    frame, buttons = boxes[0], boxes[2:]

    assert abs(frame.x + frame.width / 2 - 400) <= 1
    assert abs(frame.y + frame.height / 2 - 300) <= 1
    for box in buttons:
        assert frame.contains(box.x, box.y)
        assert frame.contains(box.x + box.width - 1, box.y + box.height - 1)
    assert [action_at(boxes, *centre(boxes, b.action)) for b in buttons] == [
        "go",
        "stay",
    ]
