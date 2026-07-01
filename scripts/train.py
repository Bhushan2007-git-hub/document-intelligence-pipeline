"""
scripts/train.py
One-command script: generate data → train all models → evaluate → save.

Usage
-----
  python scripts/train.py                     # defaults
  python scripts/train.py --samples 300       # 300 per class
  python scripts/train.py --no-generate       # skip generation, use existing data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_data import generate_dataset
from src.models.pipeline import DocumentIntelligence, LLMAssistedPipeline
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Train document intelligence pipeline")
    p.add_argument("--samples",      type=int,  default=200, help="Samples per class")
    p.add_argument("--test-size",    type=float,default=0.2,  help="Test split fraction")
    p.add_argument("--data-path",    type=str,  default="data/dataset.json")
    p.add_argument("--model-dir",    type=str,  default="models/saved")
    p.add_argument("--no-generate",  action="store_true", help="Skip data generation")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Data ─────────────────────────────────────────────────────────────
    if not args.no_generate:
        print(f"Generating {args.samples} samples per class …")
        generate_dataset(n_per_class=args.samples, out_path=args.data_path)
    else:
        print(f"Using existing dataset: {args.data_path}")

    with open(args.data_path) as f:
        raw = json.load(f)

    texts  = [d["text"]  for d in raw]
    labels = [d["label"] for d in raw]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=args.test_size, random_state=42, stratify=labels,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print("=" * 60)

    # ── 2. Train ─────────────────────────────────────────────────────────────
    di = DocumentIntelligence()
    di.fit(X_train, y_train)

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    print("\n── Results ─────────────────────────────────────────────────")
    for name, pipe in [
        ("Traditional ML",    di.trad),
        ("Transformer-Based", di.transformer),
        ("LLM-Assisted",      di.llm),
    ]:
        m = pipe.evaluate(X_test, y_test)
        print(
            f"\n{name}:\n"
            f"  Acc={m['accuracy']:.4f}  P={m['precision']:.4f}"
            f"  R={m['recall']:.4f}  F1={m['f1']:.4f}"
            f"  Infer={m['infer_ms_per_sample']:.3f}ms/sample"
        )
        for cls, cm in m["per_class"].items():
            print(f"    {cls:<30} P={cm['precision']:.3f}  R={cm['recall']:.3f}  F1={cm['f1-score']:.3f}")

    # ── 4. Save ──────────────────────────────────────────────────────────────
    di.save(args.model_dir)
    print(f"\nModels saved → {args.model_dir}/")
    print("Done. Start API with:  uvicorn src.backend.main:app --reload")


if __name__ == "__main__":
    main()
