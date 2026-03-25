"""
utils/metrics.py

Evaluation metrics for hierarchical multi-label classification.
  - mAP per level (mood / genre / sub-genre) and overall
  - Hierarchical consistency score
  - Per-class AP for rare sub-genre analysis
"""

import numpy as np
from sklearn.metrics import average_precision_score
from typing import Dict, List, Tuple


def compute_map(preds: np.ndarray, targets: np.ndarray,
                level_name: str = "all") -> float:
    """
    preds   : (N, C) float in [0,1]
    targets : (N, C) binary
    Returns mean average precision, skipping classes with no positives.
    """
    aps = []
    for c in range(targets.shape[1]):
        if targets[:, c].sum() == 0:
            continue
        try:
            ap = average_precision_score(targets[:, c], preds[:, c])
            aps.append(ap)
        except Exception:
            pass
    return float(np.mean(aps)) if aps else 0.0


def compute_per_class_ap(preds: np.ndarray, targets: np.ndarray,
                          label_names: List[str]) -> Dict[str, float]:
    result = {}
    for c, name in enumerate(label_names):
        if targets[:, c].sum() == 0:
            result[name] = float("nan")
        else:
            try:
                result[name] = average_precision_score(
                    targets[:, c], preds[:, c])
            except Exception:
                result[name] = float("nan")
    return result


def compute_consistency(preds: np.ndarray,
                         n_mood: int, n_genre: int, n_subgenre: int,
                         genre_to_mood: List[int],
                         subgenre_to_genre: List[int],
                         threshold: float = 0.5) -> Dict[str, float]:
    """
    Fraction of predictions where hierarchical consistency holds:
      - if genre[g] > thresh, then mood[genre_to_mood[g]] > thresh
      - if subgenre[s] > thresh, then genre[subgenre_to_genre[s]] > thresh
    """
    om, og = 0, n_mood
    os_ = n_mood + n_genre

    p_mood  = preds[:, om:og]     > threshold
    p_genre = preds[:, og:os_]    > threshold
    p_sub   = preds[:, os_:]      > threshold

    # genre -> mood consistency
    genre_consistent = []
    for gi, mi in enumerate(genre_to_mood):
        genre_pos = p_genre[:, gi]
        mood_pos  = p_mood[:, mi]
        # consistent if: NOT(genre=1 AND mood=0)
        cons = ~(genre_pos & ~mood_pos)
        genre_consistent.append(cons.mean())

    # subgenre -> genre consistency
    sub_consistent = []
    for si, gi in enumerate(subgenre_to_genre):
        sub_pos   = p_sub[:, si]
        genre_pos = p_genre[:, gi]
        cons = ~(sub_pos & ~genre_pos)
        sub_consistent.append(cons.mean())

    return {
        "consistency_genre_mood":    float(np.mean(genre_consistent)),
        "consistency_sub_genre":     float(np.mean(sub_consistent)),
        "consistency_overall":       float(np.mean(genre_consistent + sub_consistent)),
    }


def compute_rare_class_metrics(preds: np.ndarray, targets: np.ndarray,
                                 label_names: List[str],
                                 train_counts: Dict[str, int],
                                 rare_threshold: int = 200
                                 ) -> Tuple[float, float]:
    """
    Separate mAP for rare (< rare_threshold training samples) vs common classes.
    Returns (mAP_rare, mAP_common).
    """
    rare_aps, common_aps = [], []
    for c, name in enumerate(label_names):
        if targets[:, c].sum() == 0:
            continue
        count = train_counts.get(name, 0)
        try:
            ap = average_precision_score(targets[:, c], preds[:, c])
        except Exception:
            continue
        if count < rare_threshold:
            rare_aps.append(ap)
        else:
            common_aps.append(ap)

    return (float(np.mean(rare_aps))   if rare_aps   else 0.0,
            float(np.mean(common_aps)) if common_aps else 0.0)


def evaluate(preds: np.ndarray, targets: np.ndarray,
             n_mood: int, n_genre: int, n_subgenre: int,
             genre_to_mood: List[int],
             subgenre_to_genre: List[int],
             label_names: List[str] = None) -> Dict[str, float]:
    """
    Full evaluation suite. Returns a dict of all metrics.
    """
    om, og = 0, n_mood
    os_ = n_mood + n_genre

    results = {
        "mAP_all":      compute_map(preds, targets),
        "mAP_mood":     compute_map(preds[:, om:og],  targets[:, om:og]),
        "mAP_genre":    compute_map(preds[:, og:os_], targets[:, og:os_]),
        "mAP_subgenre": compute_map(preds[:, os_:],   targets[:, os_:]),
    }
    results.update(compute_consistency(
        preds, n_mood, n_genre, n_subgenre,
        genre_to_mood, subgenre_to_genre))

    return results


def print_results(results: Dict[str, float], prefix: str = ""):
    width = max(len(k) for k in results) + 2
    print(f"\n{'─'*50}")
    if prefix:
        print(f"  {prefix}")
    for k, v in results.items():
        print(f"  {k:{width}s}: {v:.4f}")
    print(f"{'─'*50}")
