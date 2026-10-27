"""CPU-only checks for the ontology, DAG mask, and metrics (no audio, no GPU)."""
import numpy as np
import torch

from utils.hierarchy import get_hierarchy_config
from utils.metrics import compute_consistency, compute_map

H = get_hierarchy_config()
NM, NG, NS = H["n_mood"], H["n_genre"], H["n_subgenre"]


def test_parent_indices_in_range():
    assert all(0 <= m < NM for m in H["genre_to_mood"])
    assert all(0 <= g < NG for g in H["subgenre_to_genre"])
    assert len(H["genre_to_mood"]) == NG
    assert len(H["subgenre_to_genre"]) == NS


def test_mask_shape_symmetry_and_no_mood_subgenre_edge():
    m = H["hierarchy_mask"]
    n = NM + NG + NS
    assert m.shape == (n, n)
    assert torch.equal(m, m.T)
    assert m[:NM, NM + NG:].sum() == 0  # mood <-> subgenre must go through genre


def test_mask_has_parent_edges():
    m = H["hierarchy_mask"]
    for s, g in enumerate(H["subgenre_to_genre"]):
        assert m[NM + NG + s, NM + g] == 1


def _preds(rows):
    return np.array(rows, dtype=float)


def test_consistency_detects_violation():
    n = NM + NG + NS
    ok = np.zeros((1, n))
    bad = np.zeros((1, n))
    bad[0, NM + 0] = 1.0  # genre 0 on, its mood parent off
    kw = dict(n_mood=NM, n_genre=NG, n_subgenre=NS,
              genre_to_mood=H["genre_to_mood"],
              subgenre_to_genre=H["subgenre_to_genre"])
    assert compute_consistency(ok, **kw)["consistency_overall"] == 1.0
    assert compute_consistency(bad, **kw)["consistency_overall"] < 1.0


def test_map_perfect_and_skips_empty_classes():
    targets = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert compute_map(targets.astype(float), targets) == 1.0  # class 2 has no positives
