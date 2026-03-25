"""
utils/text_embeddings.py

Generates label embeddings from a pretrained sentence transformer.
Run once, cache to disk, load during training.

Usage:
    python -m utils.text_embeddings --output embeddings/label_embs.pt
"""

import os
import argparse
import torch
import numpy as np
from utils.hierarchy import get_text_label_strings


def generate_label_embeddings(model_id: str = "sentence-transformers/all-mpnet-base-v2",
                               output_path: str = "embeddings/label_embs.pt",
                               device: str = "cpu"):
    """
    Uses sentence-transformers to embed mood, genre, and sub-genre label strings.
    Saves (mood_embs, genre_embs, subgenre_embs) as a dict of tensors.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading text LM: {model_id}")
    model = SentenceTransformer(model_id, device=device)
    model.eval()

    mood_strs, genre_strs, sub_strs = get_text_label_strings()

    print(f"Encoding {len(mood_strs)} mood labels...")
    mood_embs = model.encode(mood_strs, convert_to_tensor=True,
                              normalize_embeddings=True)

    print(f"Encoding {len(genre_strs)} genre labels...")
    genre_embs = model.encode(genre_strs, convert_to_tensor=True,
                               normalize_embeddings=True)

    print(f"Encoding {len(sub_strs)} sub-genre labels...")
    sub_embs = model.encode(sub_strs, convert_to_tensor=True,
                             normalize_embeddings=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "mood":     mood_embs.cpu(),
        "genre":    genre_embs.cpu(),
        "subgenre": sub_embs.cpu(),
        "model_id": model_id,
        "dim":      mood_embs.shape[-1],
    }, output_path)

    print(f"Saved embeddings to {output_path}")
    print(f"  mood:     {mood_embs.shape}")
    print(f"  genre:    {genre_embs.shape}")
    print(f"  subgenre: {sub_embs.shape}")
    return output_path


def load_label_embeddings(path: str):
    data = torch.load(path, map_location="cpu")
    return data["mood"], data["genre"], data["subgenre"]


def analyse_embedding_geometry(path: str):
    """
    Verify that text embeddings encode semantic relationships.
    Prints top-5 nearest neighbours for each genre.
    Useful sanity check before training.
    """
    from utils.hierarchy import GENRES, SUBGENRES
    mood_embs, genre_embs, sub_embs = load_label_embeddings(path)

    # Genre <-> sub-genre similarity
    print("\nTop-3 closest sub-genres per genre (cosine similarity):")
    sims = genre_embs @ sub_embs.T                      # (n_genre, n_sub)
    for gi, gname in enumerate(GENRES):
        topk = sims[gi].topk(3).indices.tolist()
        subs = [SUBGENRES[i] for i in topk]
        print(f"  {gname:20s} -> {subs}")

    # Mood <-> genre similarity
    print("\nTop-3 closest genres per mood:")
    sims = mood_embs @ genre_embs.T                     # (n_mood, n_genre)
    from utils.hierarchy import MOODS
    for mi, mname in enumerate(MOODS):
        topk = sims[mi].topk(3).indices.tolist()
        genres = [GENRES[i] for i in topk]
        print(f"  {mname:20s} -> {genres}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output", default="embeddings/label_embs.pt")
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args()

    path = generate_label_embeddings(args.model, args.output)
    if args.analyse:
        analyse_embedding_geometry(path)
