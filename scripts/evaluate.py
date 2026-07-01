"""
scripts/evaluate.py
Load saved models and run a full evaluation on a dataset.

Usage
-----
  python scripts/evaluate.py
  python scripts/evaluate.py --data-path data/dataset.json --model-dir models/saved
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.models.pipeline import (
    CLASSES, DocumentIntelligence, LLMAssistedPipeline,
    TraditionalMLPipeline, TransformerPipeline,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",  default="data/dataset.json")
    p.add_argument("--model-dir",  default="models/saved")
    p.add_argument("--test-size",  type=float, default=0.2)
    return p.parse_args()


def print_cm(cm, classes):
    col_w = max(len(c) for c in classes) + 2
    header = " " * col_w + "".join(f"{c[:8]:>10}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        vals = "".join(
            f"\033[92m{v:>10}\033[0m" if i == j else
            (f"\033[91m{v:>10}\033[0m" if v > 0 else f"{'0':>10}")
            for j, v in enumerate(row)
        )
        print(f"{classes[i][:col_w-1]:<{col_w}}{vals}")


def main():
    args = parse_args()

    with open(args.data_path) as f:
        raw = json.load(f)
    texts  = [d["text"]  for d in raw]
    labels = [d["label"] for d in raw]

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels,
    )
    print(f"Test samples: {len(X_test)}")

    model_path = Path(args.model_dir)
    pipes = {}
    if (model_path / "trad_model.joblib").exists():
        pipes["Traditional ML"]    = TraditionalMLPipeline.load(str(model_path / "trad_model.joblib"))
        pipes["Transformer-Based"] = TransformerPipeline.load(str(model_path / "transformer_model.joblib"))
    pipes["LLM-Assisted"] = LLMAssistedPipeline()

    for name, pipe in pipes.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print("="*60)
        m = pipe.evaluate(X_test, y_test)
        print(
            f"  Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}"
            f"  Recall={m['recall']:.4f}  F1={m['f1']:.4f}"
            f"  Infer={m['infer_ms_per_sample']:.3f}ms/sample"
        )
        print("\n  Per-class metrics:")
        for cls, cm in m["per_class"].items():
            print(f"    {cls:<30} P={cm['precision']:.3f}  R={cm['recall']:.3f}  F1={cm['f1-score']:.3f}")

        preds = pipe.predict(X_test)
        cm = confusion_matrix(y_test, preds, labels=CLASSES)
        print("\n  Confusion matrix (rows=true, cols=pred):")
        print_cm(cm.tolist(), CLASSES)


if __name__ == "__main__":
    main()
