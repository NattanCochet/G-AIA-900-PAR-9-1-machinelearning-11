"""T3-03 split logic on fake MUSDB clip names (no audio)."""

import random

import pytest

from task3_uvr.check_data import artist_of, artist_overlap
from task3_uvr.make_split import group_by_artist, split_by_artist

# 25 clips, 14 artists with 1 to 5 clips each.
CLIPS = (
    [f"Solo{i} - Song" for i in range(8)]
    + [f"Duo{i} - Song {j}" for i in range(3) for j in range(2)]
    + [f"Big Band - Track {j}" for j in range(5)]
    + [f"Trio - Part {j}" for j in range(3)]
    + ["Guest - One", "Guest feat. Someone - Two", "Guest (Remix) - Three"]
)


def test_artist_of_drops_guests_and_remixers():
    assert artist_of("Music Delta - Beatles") == "Music Delta"
    assert artist_of("Triviul feat. The Fiend - Widow") == "Triviul"
    assert artist_of("The Easton Ellises (Baumi) - SDRNR") == "The Easton Ellises"
    assert artist_of("Jokers, Jacks & Kings - Sea Of Leaves") == "Jokers, Jacks & Kings"
    assert artist_of("A - B - C") == "A"


def test_group_by_artist():
    groups = group_by_artist(CLIPS)
    assert len(groups) == 14
    assert groups["Guest"] == sorted(
        ["Guest - One", "Guest feat. Someone - Two", "Guest (Remix) - Three"]
    )
    assert sum(len(g) for g in groups.values()) == len(CLIPS)


@pytest.mark.parametrize("seed", range(20))
def test_split_has_no_artist_overlap_and_keeps_every_clip(seed):
    train, validation = split_by_artist(CLIPS, fraction=0.15, seed=seed)
    assert sorted(train + validation) == sorted(CLIPS)
    assert not set(train) & set(validation)
    assert not {artist_of(c) for c in train} & {artist_of(c) for c in validation}
    assert 0 < len(validation) <= round(0.15 * len(CLIPS))


def test_split_is_deterministic_and_order_independent():
    shuffled = CLIPS[:]
    random.Random(1).shuffle(shuffled)
    assert split_by_artist(CLIPS, 0.15, seed=3) == split_by_artist(shuffled, 0.15, 3)


def test_seed_changes_the_split():
    splits = {tuple(split_by_artist(CLIPS, 0.15, seed=s)[1]) for s in range(10)}
    assert len(splits) > 1


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        split_by_artist(CLIPS, fraction=0.0, seed=0)
    with pytest.raises(ValueError):
        split_by_artist(CLIPS + CLIPS[:1], fraction=0.15, seed=0)


def test_artist_overlap():
    overlap = artist_overlap(
        ["Triviul - Dorothy", "Leaf - Wicked"],
        ["Triviul feat. The Fiend - Widow", "Zeno - Signs"],
    )
    assert list(overlap["shared_artists"]) == ["Triviul"]
    assert overlap["shared_artists"]["Triviul"]["test"] == [
        "Triviul feat. The Fiend - Widow"
    ]
