from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swath.data.corpora import Corpus, get_corpus, names, register
from swath.data.dataset import Sample, build_label_map


def test_loveda_is_registered():
    corpus = get_corpus("loveda")
    assert corpus.default_task == "landcover"
    assert corpus.domains == ("Rural", "Urban")
    assert "Train" in corpus.splits
    assert corpus.labels() is not None


def test_unknown_corpus_lists_the_known_ones():
    with pytest.raises(KeyError, match="loveda"):
        get_corpus("nowhere")


def test_registering_a_duplicate_fails():
    with pytest.raises(ValueError, match="already registered"):
        register(get_corpus("loveda"))


def _toy(samples: list[Sample], recorded: dict) -> Corpus:
    def discover(root, split, domains=None):
        recorded["root"] = Path(root)
        recorded["split"] = split
        recorded["domains"] = domains
        return samples

    return Corpus(
        name="toy",
        title="Toy",
        default_task="buildings",
        splits=("Train", "Val"),
        discover=discover,
        label_map=lambda: build_label_map({1: 0, 2: 1}),
        domains=("A", "B"),
    )


def test_samples_pass_the_domain_through(samples):
    recorded: dict = {}
    corpus = _toy(list(samples), recorded)

    corpus.samples("root", "Val", "A")
    assert recorded["split"] == "Val"
    assert recorded["domains"] == ("A",)

    corpus.samples("root", "Train")
    assert recorded["domains"] == ("A", "B")


def test_unknown_split_is_reported(samples):
    corpus = _toy(list(samples), {})
    with pytest.raises(ValueError, match="no split 'Test'"):
        corpus.samples("root", "Test")


def test_unknown_domain_is_reported(samples):
    corpus = _toy(list(samples), {})
    with pytest.raises(ValueError, match="no domain 'C'"):
        corpus.samples("root", "Train", "C")


def test_a_corpus_without_domains_is_called_without_them(samples):
    seen: dict = {}

    def discover(root, split):
        seen["called"] = (root, split)
        return list(samples)

    corpus = Corpus(
        name="flat",
        title="Flat",
        default_task="buildings",
        splits=("Train",),
        discover=discover,
    )
    assert corpus.samples("root", "Train") == list(samples)
    assert seen["called"] == ("root", "Train")
    assert corpus.labels() is None


def test_label_map_is_a_lookup_table():
    table = _toy([], {}).labels()
    assert isinstance(table, np.ndarray)
    assert table.shape == (256,)


def test_names_are_sorted():
    assert names() == sorted(names())
