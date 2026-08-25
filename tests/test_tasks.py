from __future__ import annotations

import pytest

from swath.tasks import TASKS, Task, get_task


def test_registry_exposes_the_built_in_tasks():
    names = TASKS.names()
    assert "landcover" in names
    assert "buildings" in names


def test_landcover_classes_and_palette_line_up():
    task = get_task("landcover")
    assert task.num_classes == 7
    assert len(task.palette) == 7
    assert task.classes[0] == "background"


def test_unknown_task_lists_the_known_ones():
    with pytest.raises(KeyError, match="landcover"):
        get_task("does-not-exist")


def test_palette_length_is_validated():
    with pytest.raises(ValueError, match="palette"):
        Task(
            name="broken",
            title="Broken",
            description="",
            classes=("a", "b"),
            palette=((0, 0, 0),),
        )


def test_normalisation_must_match_the_band_count():
    with pytest.raises(ValueError, match="mean/std"):
        Task(
            name="broken",
            title="Broken",
            description="",
            classes=("a", "b"),
            palette=((0, 0, 0), (1, 1, 1)),
            in_channels=6,
        )


def test_registering_a_duplicate_name_fails():
    with pytest.raises(ValueError, match="already registered"):
        TASKS.register(get_task("landcover"))


def test_tasks_are_hashable_and_comparable():
    assert get_task("landcover") == get_task("landcover")
    assert len({get_task("landcover"), get_task("landcover")}) == 1
