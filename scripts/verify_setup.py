"""
scripts/verify_setup.py

Run this first to check your environment has everything needed
before starting training.

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --data_root data/mtg_jamendo
"""

import sys
import os
import argparse


def check(label, fn):
    try:
        result = fn()
        status = result if isinstance(result, str) else "OK"
        print(f"  [OK]     {label}: {status}")
        return True
    except Exception as e:
        print(f"  [FAIL]   {label}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/mtg_jamendo")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  H-ATGNN environment check")
    print("="*55)

    failures = []

    # ── Python version ──
    print("\n[Python]")
    v = sys.version_info
    ok = check("Python version",
               lambda: f"{v.major}.{v.minor}.{v.micro} {'OK' if v >= (3,8) else '(need >= 3.8)'}")
    if not ok: failures.append("python")

    # ── Core packages ──
    print("\n[Core packages]")
    packages = [
        ("torch",           lambda: __import__("torch").__version__),
        ("torchaudio",      lambda: __import__("torchaudio").__version__),
        ("torch_geometric", lambda: __import__("torch_geometric").__version__),
        ("torch_scatter",   lambda: __import__("torch_scatter").__version__),
        ("numpy",           lambda: __import__("numpy").__version__),
        ("sklearn",         lambda: __import__("sklearn").__version__),
        ("tqdm",            lambda: __import__("tqdm").__version__),
    ]
    for name, fn in packages:
        if not check(name, fn):
            failures.append(name)

    # ── Optional packages (for text init + CLAP) ──
    print("\n[Optional packages]")
    optional = [
        ("sentence_transformers",
         lambda: __import__("sentence_transformers").__version__,
         "needed for --text_init"),
        ("transformers",
         lambda: __import__("transformers").__version__,
         "needed for --clap (transformers backend)"),
        ("laion_clap",
         lambda: "installed",
         "alternative CLAP backend (pip install laion-clap)"),
    ]
    for name, fn, note in optional:
        try:
            result = fn()
            print(f"  [OK]     {name}: {result}")
        except Exception:
            print(f"  [WARN]   {name}: not installed  ({note})")

    # ── CUDA ──
    print("\n[Hardware]")
    check("CUDA available", lambda: (
        f"Yes — {__import__('torch').cuda.get_device_name(0)}"
        if __import__('torch').cuda.is_available()
        else "No — will train on CPU (slow)"
    ))

    # ── Data ──
    print("\n[Data]")
    from pathlib import Path

    def check_data():
        root = args.data_root
        if not os.path.exists(root):
            raise FileNotFoundError(
                f"{root} not found. Run:\n"
                "    python scripts/download_mtg_jamendo.py "
                f"--output {root} --chunks 0"
            )
        n_audio = sum(1 for _ in Path(root).rglob("*.mp3"))
        n_clap  = sum(1 for _ in Path(root).rglob("*.clap.pt"))
        splits  = os.path.join(root, "splits", "split-0")
        has_splits = all(
            os.path.exists(os.path.join(splits, f))
            for f in ["train.tsv", "validation.tsv", "test.tsv"]
        )
        return (f"{n_audio} mp3 files, {n_clap} CLAP embeddings, "
                f"splits={'OK' if has_splits else 'MISSING'}")

    if not check(f"data at {args.data_root}", check_data):
        failures.append("data")

    # ── Label embeddings ──
    print("\n[Label embeddings]")
    def check_emb():
        path = "embeddings/label_embs.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run:\n"
                "    python -m utils.text_embeddings --output embeddings/label_embs.pt"
            )
        import torch
        data = torch.load(path, map_location="cpu")
        return (f"mood={data['mood'].shape}, genre={data['genre'].shape}, "
                f"subgenre={data['subgenre'].shape}")

    check("label_embs.pt", check_emb)  # warning only, not a hard failure

    # ── Project imports ──
    print("\n[Project imports]")
    project_imports = [
        ("configs.default",   lambda: __import__("configs.default", fromlist=["HATGNNConfig"])),
        ("utils.hierarchy",   lambda: __import__("utils.hierarchy",  fromlist=["get_hierarchy_config"])),
        ("utils.metrics",     lambda: __import__("utils.metrics",    fromlist=["evaluate"])),
        ("models.hatgnn",     lambda: __import__("models.hatgnn",    fromlist=["HATGNN"])),
    ]
    for name, fn in project_imports:
        if not check(name, lambda f=fn: f() and "OK"):
            failures.append(name)

    # ── Summary ──
    print("\n" + "="*55)
    if not failures:
        print("  All checks passed. Ready to train.")
        print("\n  Quick start:")
        print("    python train.py --text_init --hierarchy --clap")
        print("    python train.py --run_all_ablations")
    else:
        hard = [f for f in failures
                if f not in ("data", "sentence_transformers",
                             "transformers", "laion_clap")]
        if hard:
            print(f"  FAILED checks: {', '.join(hard)}")
            print("  Fix these before training.")
        else:
            print(f"  Minor issues: {', '.join(failures)}")
            print("  Core deps OK. Some features may be unavailable.")
    print("="*55 + "\n")


if __name__ == "__main__":
    # Make sure project root is on path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
