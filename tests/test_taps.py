"""A tap on the kiosk plays the bird under it: the page knows where each bird
landed, and the detector hands over its surest call of that bird. The routes
are tested with the others, in test_server.py."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from fugleramme import api, fake, names
from fugleramme.render import collage
from fugleramme.render.collage import NO_BIRDS, Birds, draw_collage, whole_page
from fugleramme.source import Detection

BLACKBIRD = "Turdus merula"


def _drawn(page: Image.Image, birds: Birds) -> tuple[np.ndarray, np.ndarray]:
    """Per hit cell, whether the page has ink at its centre and whether the map
    has a bird there."""
    rows, cols = birds.grid.shape
    grey = np.asarray(page.convert("L"))
    ys = np.minimum((np.arange(rows) + 0.5) * page.height / rows, page.height - 1).astype(int)
    xs = np.minimum((np.arange(cols) + 0.5) * page.width / cols, page.width - 1).astype(int)
    return grey[np.ix_(ys, xs)] < 100, birds.grid > 0


def test_the_map_puts_each_bird_where_it_is_drawn(crowded):
    entries = crowded(5)
    page, birds = draw_collage(entries, (640, 400), show_names=False)
    ink, mapped = _drawn(page, birds)
    assert not (ink & ~mapped).any()
    assert mapped.sum() < 1.3 * ink.sum()  # the plate's paper halo answers too, no more
    assert set(np.unique(birds.grid)) == set(range(len(entries) + 1))
    assert birds.names == tuple(name for name, _ in entries)


def test_a_tap_on_a_name_is_a_tap_on_its_bird(crowded):
    page, birds = draw_collage(crowded(3), (640, 400))
    ink, mapped = _drawn(page, birds)
    assert not (ink & ~mapped).any()


def test_a_keyed_page_maps_its_birds(crowded):
    """Each bird and its number; the key's list of names is not a bird."""
    entries = crowded(3)
    _page, birds = draw_collage(entries, (640, 400), name_key=True)
    assert set(np.unique(birds.grid)) == set(range(len(entries) + 1))


def test_a_tap_reads_the_bird_under_it_and_paper_reads_none(crowded):
    _page, birds = draw_collage(crowded(3), (640, 400), show_names=False)
    rows, cols = birds.grid.shape
    row, col = (int(v) for v in np.argwhere(birds.grid == 2)[0])
    assert birds.at((col + 0.5) / cols, (row + 0.5) / rows) == birds.names[1]

    bare = np.argwhere(birds.grid == 0)
    far = [
        (r, c)
        for r, c in bare
        if not birds.grid[max(0, r - 4) : r + 5, max(0, c - 4) : c + 5].any()
    ]
    if far:
        r, c = far[0]
        assert birds.at((c + 0.5) / cols, (r + 0.5) / rows) is None


def test_a_tap_just_off_a_bird_still_finds_it():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 5] = 1
    birds = Birds(("Genus species",), grid)
    assert birds.at(0.75, 0.55) == "Genus species"  # two cells off
    assert birds.at(0.95, 0.05) is None
    assert birds.at(1.5, 0.5) is None


def test_a_page_of_one_bird_is_that_bird_everywhere_and_a_bare_page_none():
    assert whole_page(BLACKBIRD).at(0.0, 1.0) == BLACKBIRD
    assert NO_BIRDS.at(0.5, 0.5) is None
    assert draw_collage([], (300, 200))[1] is NO_BIRDS


def test_render_collage_is_the_drawn_page(crowded):
    collage._layouts.clear()
    drawn = np.asarray(draw_collage(crowded(4), (500, 400))[0])
    assert np.array_equal(np.asarray(collage.render_collage(crowded(4), (500, 400))), drawn)


def test_a_reclassified_bird_is_asked_for_under_both_its_names():
    assert names.spellings("Corvus monedula") == ("Coloeus monedula", "Corvus monedula")
    assert names.spellings("Coloeus monedula") == ("Coloeus monedula", "Corvus monedula")
    assert names.spellings(BLACKBIRD) == (BLACKBIRD,)


def test_the_best_call_is_the_surest_that_is_not_a_false_positive(detector):
    rows = fake.generate(60, seed=3)
    blackbirds = [r for r in rows if r.scientific_name == BLACKBIRD]
    assert blackbirds
    rows = [*rows, replace(blackbirds[0], id=10_000, confidence=1.0, false_positive=True)]
    url, _httpd = detector(rows=rows)
    source = api.ApiSource(url)

    call = source.best_call(BLACKBIRD)

    real = [r for r in blackbirds if not r.false_positive]
    assert isinstance(call, Detection)
    assert call.id == max(real, key=lambda r: r.confidence).id
    assert source.best_call("Genus nonexistens") is None


def test_a_clip_comes_through_and_an_unknown_one_is_none(source):
    detector = source(count=10, seed=1)
    call = detector.recent(1)[0]
    content_type, body = detector.clip(call.id) or ("", b"")
    assert content_type == "audio/wav" and body.startswith(b"RIFF")
    assert detector.clip(999_999) is None
