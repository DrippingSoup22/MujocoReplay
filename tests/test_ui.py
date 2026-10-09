"""Tests for the side panel and the tabs: where they go and what a click does."""

from mujoco_replay.ui import (
    Buttons,
    Panel,
    Section,
    Stepper,
    TabBar,
    Title,
    Toggles,
    action_at,
    dialog,
    tab_names,
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


def test_tabs_sit_in_a_bar_along_the_top_each_with_a_button_that_closes_it():
    long = "start" + "-" * 40 + "end"
    boxes = TabBar().layout(["early", "late", long], 1, 300, 800, 600, LINE, measure)
    bar, placed = boxes[0], boxes[1:]

    assert (bar.x, bar.y + bar.height, bar.width) == (300, 600, 800)
    for box in placed:
        assert bar.contains(box.x, box.y)
        assert bar.contains(box.x + box.width - 1, box.y + box.height - 1)
    assert [action_at(boxes, *centre(boxes, box.action)) for box in placed] == [
        "tab 0",
        "close tab 0",
        "tab 1",
        "close tab 1",
        "tab 2",
        "close tab 2",
        "open",
    ]
    assert [box.kind for box in placed if box.action.endswith("1")] == ["lit"] * 2
    cut = placed[4].text  # a long name keeps its start and its end
    assert cut.startswith("start") and "..." in cut and cut.endswith("end")


def test_tabs_that_do_not_fit_scroll_and_the_shown_tab_comes_into_view():
    bar = TabBar()
    names = [f"cycle_{index:04d}" for index in range(30)]

    def shown(current: int) -> list[int]:
        boxes = bar.layout(names, current, 300, 600, 600, LINE, measure)
        assert {"tabs left", "tabs right"} <= {box.action for box in boxes}
        return [int(box.action[4:]) for box in boxes if box.action.startswith("tab ")]

    assert 25 in shown(25)
    bar.scroll_by(-100)
    assert shown(25)[0] == 0  # moved away from the shown tab, as asked
    bar.scroll_by(100)
    assert shown(25)[-1] == 29  # and no further than the last
    assert 3 in shown(3)
    bar.scroll_by(10)
    boxes = bar.layout(names, 3, 300, 500, 600, LINE, measure)  # narrower
    assert "tab 3" in {box.action for box in boxes}


def test_a_narrow_bar_keeps_its_tabs_inside_and_big_enough_to_click():
    names = [f"cycle_{index:04d}" for index in range(8)]
    for width in range(0, 600, 7):
        boxes = TabBar().layout(names, 5, 300, width, 600, LINE, measure)

        for box in boxes[1:]:
            assert 300 <= box.x and box.x + box.width <= 300 + width
        tabs = [box for box in boxes if box.action.startswith("tab ")]
        assert all(box.width >= 2 * LINE for box in tabs)


def test_tabs_are_named_by_their_files_and_same_names_by_a_folder_too():
    paths = [
        "/runs/a/recordings/cycle_1.npz",
        "/runs/b/recordings/cycle_1.npz",
        "/runs/a/recordings/cycle_2.npz",
        None,
    ]
    clashing = ["/e1/runA/rec/c.npz", "/e2/runA/rec/c.npz", "/e1/runB/rec/c.npz"]

    names = tab_names(paths, ["", "", "", "Café run"])
    three = tab_names(clashing, ["", "", ""])

    assert names == ["cycle_1 (a)", "cycle_1 (b)", "cycle_2", "Cafe run"]
    assert three == ["c (e1/runA)", "c (e2/runA)", "c (e1/runB)"]
