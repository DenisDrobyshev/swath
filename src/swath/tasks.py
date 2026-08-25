"""Task registry.

A *task* is everything that distinguishes one segmentation problem from
another: the class list, the colour each class is drawn with, the number of
input channels and the normalisation statistics. The model code and the
service code stay identical across tasks, which is what makes it cheap to add
a new one — buildings, roads, flood extent — without touching anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IGNORE_INDEX = 255
"""Label value excluded from both the loss and the metrics."""


@dataclass(frozen=True)
class Task:
    """Description of a single segmentation problem."""

    name: str
    title: str
    description: str
    classes: tuple[str, ...]
    palette: tuple[tuple[int, int, int], ...]
    in_channels: int = 3
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    source: str = ""
    license: str = ""

    def __post_init__(self) -> None:
        if len(self.classes) != len(self.palette):
            raise ValueError(
                f"task {self.name!r}: {len(self.classes)} classes but "
                f"{len(self.palette)} palette entries"
            )
        if len(self.mean) != self.in_channels or len(self.std) != self.in_channels:
            raise ValueError(f"task {self.name!r}: mean/std must have {self.in_channels} entries")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def color_of(self, index: int) -> tuple[int, int, int]:
        return self.palette[index]


@dataclass(frozen=True)
class _Registry:
    tasks: dict[str, Task] = field(default_factory=dict)

    def register(self, task: Task) -> Task:
        if task.name in self.tasks:
            raise ValueError(f"task {task.name!r} is already registered")
        self.tasks[task.name] = task
        return task

    def __getitem__(self, name: str) -> Task:
        try:
            return self.tasks[name]
        except KeyError:
            known = ", ".join(sorted(self.tasks)) or "none"
            raise KeyError(f"unknown task {name!r}; registered tasks: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.tasks

    def __iter__(self):
        return iter(self.tasks.values())

    def names(self) -> list[str]:
        return sorted(self.tasks)


TASKS = _Registry()


TASKS.register(
    Task(
        name="landcover",
        title="Land cover (7 classes)",
        description=(
            "Land-cover types on 0.3 m aerial imagery: background, building, road, "
            "water, barren, forest and agriculture."
        ),
        classes=(
            "background",
            "building",
            "road",
            "water",
            "barren",
            "forest",
            "agriculture",
        ),
        palette=(
            (255, 255, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 0, 255),
            (159, 129, 183),
            (0, 255, 0),
            (255, 195, 128),
        ),
        source="LoveDA (Wang et al., 2021)",
        license="CC BY 4.0",
    )
)


TASKS.register(
    Task(
        name="buildings",
        title="Building footprints",
        description="Binary building footprint extraction from aerial or satellite tiles.",
        classes=("background", "building"),
        palette=((255, 255, 255), (222, 45, 38)),
    )
)


TASKS.register(
    Task(
        name="roads",
        title="Road network",
        description="Binary road-surface extraction from aerial or satellite tiles.",
        classes=("background", "road"),
        palette=((255, 255, 255), (250, 159, 30)),
    )
)


TASKS.register(
    Task(
        name="greenery",
        title="Urban greenery",
        description="Vegetation cover in the urban fabric: trees, grass and everything else.",
        classes=("other", "tree", "grass"),
        palette=((235, 235, 235), (13, 105, 51), (124, 205, 80)),
    )
)


TASKS.register(
    Task(
        name="flood",
        title="Flood extent",
        description="Flooded versus dry surface on post-disaster drone imagery.",
        classes=("dry", "flooded", "water"),
        palette=((235, 235, 235), (196, 48, 145), (28, 106, 196)),
    )
)


def get_task(name: str) -> Task:
    """Look a task up by name, raising a helpful error when it is unknown."""
    return TASKS[name]
