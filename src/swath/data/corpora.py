"""Registry of datasets the command line can work with.

Adding a corpus should be a matter of describing where its files are and how its
labels map onto a task — not of editing the CLI, the trainer and the evaluator.
A :class:`Corpus` holds exactly those two things plus the split names, and every
command reaches its data through this registry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swath.data.dataset import Sample


@dataclass(frozen=True)
class Corpus:
    """How to find and interpret one dataset on disk.

    Args:
        name: Identifier used on the command line.
        title: Human-readable name.
        default_task: Task from the registry these labels belong to.
        splits: Split names, in the order a user would expect to see them.
        train_split, val_split: Which of those splits training uses.
        discover: ``(root, split, domains) -> [Sample, ...]``.
        label_map: Optional 256-entry lookup applied to every raw label map.
        prepare: Optional ``(raw_dir, out_dir) -> {split: count}`` unpacking step.
        domains: Sub-collections a split can be filtered to, when the corpus has any.
        source: Where the data comes from, for the command line to print.
    """

    name: str
    title: str
    default_task: str
    splits: tuple[str, ...]
    discover: Callable[..., list[Sample]]
    train_split: str = "Train"
    val_split: str = "Val"
    label_map: Callable[[], np.ndarray | None] | None = None
    prepare: Callable[[Path, Path], dict[str, int]] | None = None
    domains: tuple[str, ...] = ()
    source: str = ""

    def labels(self) -> np.ndarray | None:
        return self.label_map() if self.label_map else None

    def samples(self, root: Path | str, split: str, domain: str = "both") -> list[Sample]:
        """Collect one split, optionally narrowed to a single domain."""
        if split not in self.splits:
            raise ValueError(
                f"{self.name} has no split {split!r}; expected one of {self.splits}"
            )
        if domain != "both" and self.domains and domain not in self.domains:
            raise ValueError(
                f"{self.name} has no domain {domain!r}; expected one of {self.domains}"
            )
        if not self.domains:
            return self.discover(root, split)
        chosen = self.domains if domain == "both" else (domain,)
        return self.discover(root, split, domains=chosen)


_REGISTRY: dict[str, Corpus] = {}


def register(corpus: Corpus) -> Corpus:
    if corpus.name in _REGISTRY:
        raise ValueError(f"corpus {corpus.name!r} is already registered")
    _REGISTRY[corpus.name] = corpus
    return corpus


def get_corpus(name: str) -> Corpus:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown dataset {name!r}; registered: {known}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_corpora() -> Sequence[Corpus]:
    return [_REGISTRY[name] for name in names()]


def _register_builtin() -> None:
    from swath.data import loveda

    register(
        Corpus(
            name="loveda",
            title="LoveDA land cover",
            default_task="landcover",
            splits=loveda.SPLITS,
            discover=loveda.discover,
            label_map=loveda.label_map,
            prepare=loveda.prepare,
            domains=loveda.DOMAINS,
            source=loveda.ZENODO_RECORD,
        )
    )


_register_builtin()


__all__ = ["Corpus", "all_corpora", "get_corpus", "names", "register"]
