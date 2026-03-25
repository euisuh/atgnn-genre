"""
utils/hierarchy.py

Defines the mood->genre->sub-genre ontology for MTG-Jamendo
and builds the hierarchy mask used in H-LLG.

Structure (condensed from MTG-Jamendo + MusicBrainz genre taxonomy):
  Mood (8)  ->  Genre (15)  ->  Sub-genre (50)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


# ── Ontology ─────────────────────────────────────────────────────────────────

MOODS: List[str] = [
    "joyful",       # 0
    "tense",        # 1
    "melancholic",  # 2
    "energetic",    # 3
    "calm",         # 4
    "dark",         # 5
    "romantic",     # 6
    "aggressive",   # 7
]

GENRES: List[str] = [
    "jazz",         # 0  -> joyful, tense
    "blues",        # 1  -> melancholic, dark
    "classical",    # 2  -> calm, romantic
    "rock",         # 3  -> energetic, aggressive
    "metal",        # 4  -> aggressive, dark
    "electronic",   # 5  -> energetic, tense
    "hip-hop",      # 6  -> energetic, aggressive
    "r&b",          # 7  -> joyful, romantic
    "pop",          # 8  -> joyful, energetic
    "folk",         # 9  -> calm, melancholic
    "country",      # 10 -> joyful, calm
    "latin",        # 11 -> joyful, energetic
    "soul",         # 12 -> romantic, melancholic
    "ambient",      # 13 -> calm, dark
    "world",        # 14 -> joyful, calm
]

SUBGENRES: List[str] = [
    # Jazz (0)
    "bebop", "post-bop", "cool jazz", "free jazz",
    # Blues (1)
    "delta blues", "chicago blues", "electric blues",
    # Classical (2)
    "baroque", "romantic classical", "contemporary classical", "opera",
    # Rock (3)
    "classic rock", "indie rock", "alternative rock", "progressive rock",
    # Metal (4)
    "heavy metal", "death metal", "black metal", "doom metal",
    # Electronic (5)
    "techno", "house", "ambient electronic", "drum and bass",
    # Hip-Hop (6)
    "old school hip-hop", "trap", "conscious hip-hop",
    # R&B (7)
    "neo-soul", "contemporary r&b", "funk",
    # Pop (8)
    "synth-pop", "indie pop", "dance pop",
    # Folk (9)
    "acoustic folk", "singer-songwriter", "freak folk",
    # Country (10)
    "traditional country", "alt-country",
    # Latin (11)
    "salsa", "bossa nova", "reggaeton",
    # Soul (12)
    "classic soul", "deep soul",
    # Ambient (13)
    "dark ambient", "drone", "new age",
    # World (14)
    "afrobeat", "celtic", "flamenco",
]

# genre_index -> [mood_indices]  (multi-mood allowed, use primary for parent mapping)
GENRE_MOOD_MAP: Dict[int, List[int]] = {
    0:  [0, 1],   # jazz -> joyful, tense
    1:  [2, 5],   # blues -> melancholic, dark
    2:  [4, 6],   # classical -> calm, romantic
    3:  [3, 7],   # rock -> energetic, aggressive
    4:  [7, 5],   # metal -> aggressive, dark
    5:  [3, 1],   # electronic -> energetic, tense
    6:  [3, 7],   # hip-hop -> energetic, aggressive
    7:  [0, 6],   # r&b -> joyful, romantic
    8:  [0, 3],   # pop -> joyful, energetic
    9:  [4, 2],   # folk -> calm, melancholic
    10: [0, 4],   # country -> joyful, calm
    11: [0, 3],   # latin -> joyful, energetic
    12: [6, 2],   # soul -> romantic, melancholic
    13: [4, 5],   # ambient -> calm, dark
    14: [0, 4],   # world -> joyful, calm
}

# subgenre_index -> genre_index
SUBGENRE_GENRE_MAP: List[int] = [
    # Jazz (genre 0)
    0, 0, 0, 0,
    # Blues (genre 1)
    1, 1, 1,
    # Classical (genre 2)
    2, 2, 2, 2,
    # Rock (genre 3)
    3, 3, 3, 3,
    # Metal (genre 4)
    4, 4, 4, 4,
    # Electronic (genre 5)
    5, 5, 5, 5,
    # Hip-Hop (genre 6)
    6, 6, 6,
    # R&B (genre 7)
    7, 7, 7,
    # Pop (genre 8)
    8, 8, 8,
    # Folk (genre 9)
    9, 9, 9,
    # Country (genre 10)
    10, 10,
    # Latin (genre 11)
    11, 11, 11,
    # Soul (genre 12)
    12, 12,
    # Ambient (genre 13)
    13, 13, 13,
    # World (genre 14)
    14, 14, 14,
]

# Primary mood parent for each genre (first in list)
GENRE_TO_MOOD_PRIMARY: List[int] = [
    GENRE_MOOD_MAP[g][0] for g in range(len(GENRES))
]


def build_hierarchy_mask(n_mood: int, n_genre: int, n_subgenre: int,
                          genre_to_mood: List[int],
                          subgenre_to_genre: List[int]) -> torch.Tensor:
    """
    Returns (n_total, n_total) binary mask where 1 means the edge is allowed.
    Layout: [mood (0..n_mood), genre (n_mood..n_mood+n_genre),
             subgenre (n_mood+n_genre..)]

    Allowed edges:
      - within mood-mood: all pairs (fully connected)
      - within genre-genre: all pairs
      - within subgenre-subgenre: all pairs
      - genre -> its mood parent (bidirectional for info flow)
      - subgenre -> its genre parent (bidirectional)
    Forbidden:
      - mood -> subgenre direct (must go through genre)
    """
    n = n_mood + n_genre + n_subgenre
    mask = torch.zeros(n, n)

    om, og, os_ = 0, n_mood, n_mood + n_genre

    # Within-level: fully connected
    mask[om:og, om:og] = 1
    mask[og:os_, og:os_] = 1
    mask[os_:, os_:] = 1

    # Genre <-> mood parent edges
    for gi, mi in enumerate(genre_to_mood):
        mask[og + gi, om + mi] = 1
        mask[om + mi, og + gi] = 1

    # Subgenre <-> genre parent edges
    for si, gi in enumerate(subgenre_to_genre):
        mask[os_ + si, og + gi] = 1
        mask[og + gi, os_ + si] = 1

    return mask


def get_label_names() -> Tuple[List[str], List[str], List[str]]:
    return MOODS, GENRES, SUBGENRES


def get_hierarchy_config():
    """Returns all hierarchy info needed to build HATGNNConfig."""
    n_mood, n_genre, n_sub = len(MOODS), len(GENRES), len(SUBGENRES)
    mask = build_hierarchy_mask(n_mood, n_genre, n_sub,
                                 GENRE_TO_MOOD_PRIMARY,
                                 SUBGENRE_GENRE_MAP)
    return dict(
        n_mood=n_mood,
        n_genre=n_genre,
        n_subgenre=n_sub,
        hierarchy_mask=mask,
        genre_to_mood=GENRE_TO_MOOD_PRIMARY,
        subgenre_to_genre=SUBGENRE_GENRE_MAP,
    )


def get_text_label_strings() -> Tuple[List[str], List[str], List[str]]:
    """
    Returns richer text descriptions for text LM embedding.
    Richer strings give better semantic geometry.
    """
    mood_strs = [
        "joyful upbeat happy music",
        "tense anxious suspenseful music",
        "melancholic sad nostalgic music",
        "energetic fast driving music",
        "calm peaceful relaxing music",
        "dark ominous brooding music",
        "romantic loving tender music",
        "aggressive intense angry music",
    ]
    genre_strs = [f"{g} music genre" for g in GENRES]
    sub_strs   = [f"{s} music subgenre" for s in SUBGENRES]
    return mood_strs, genre_strs, sub_strs
