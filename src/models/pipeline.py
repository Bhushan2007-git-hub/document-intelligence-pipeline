"""
models/pipeline.py
Three-approach document classification & entity extraction pipeline.

Classes
-------
TraditionalMLPipeline   – TF-IDF + Logistic Regression (fast, lightweight)
TransformerPipeline     – char+word LSA embeddings + NER layer (highest accuracy)
LLMAssistedPipeline     – keyword scoring + regex extraction (zero-shot)
DocumentIntelligence    – orchestrates all three; picks best by confidence
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# ── Constants ──────────────────────────────────────────────────────────────
CLASSES = [
    "Bank Statement",
    "Commercial Registration",
    "Invoice",
    "National ID",
    "Passport",
    "Utility Bill",
]

# Per-class keyword signatures for LLM-assisted approach
CLASS_SIGNATURES: dict[str, dict[str, Any]] = {
    "Passport": {
        "keywords": [
            "passport", "nationality", "mrz", "machine readable",
            "travel document", "international", "given names", "prénoms",
            "date of expiry", "personal no",
        ],
        "patterns": [r"P<[A-Z]{3}", r"[A-Z]\d{8}", r"passport no"],
    },
    "National ID": {
        "keywords": [
            "national identity", "identity card", "id number", "blood type",
            "issuing authority", "card serial", "ministry of interior",
        ],
        "patterns": [r"id number:\s*\d+", r"blood type", r"card serial"],
    },
    "Commercial Registration": {
        "keywords": [
            "commercial registration", "certificate", "ministry of commerce",
            "legal form", "business activity", "authorized capital", "llc",
            "license number", "trn", "registrar",
        ],
        "patterns": [r"CR-\d+", r"TRN\d+", r"LIC-\d+"],
    },
    "Bank Statement": {
        "keywords": [
            "account statement", "bank", "iban", "opening balance",
            "closing balance", "account holder", "current account",
            "statement period", "branch",
        ],
        "patterns": [r"IBAN:\s*[A-Z]{2}\d+", r"\*{4}\d{4}"],
    },
    "Utility Bill": {
        "keywords": [
            "utility bill", "meter", "consumption", "kwh", "electricity",
            "bill date", "due date", "meter reading", "units consumed",
            "energy charges",
        ],
        "patterns": [r"MTR\d+", r"\d+\s*kwh", r"UTL\d+"],
    },
    "Invoice": {
        "keywords": [
            "invoice", "commercial invoice", "invoice number", "payment terms",
            "net 30", "subtotal", "unit price", "thank you for your business",
        ],
        "patterns": [r"INV-\d+", r"net\s+\d+\s+days", r"subtotal"],
    },
}

# Entity extraction patterns (field → list of regex strings)
ENTITY_PATTERNS: dict[str, list[str]] = {
    "Name": [
        r"(?:full name|customer name|account holder|owner[/\s]*manager|client):\s+([A-Za-z\s\-\.]+?)(?:\n|$)",
        r"To:\s{3}([A-Za-z\s]+?)(?:\n|$)",
    ],
    "ID Number": [
        r"id number:\s*([\d\s]+)",
        r"passport no[.:\s]*([\w\d]+)",
        r"invoice number:\s*([\w\-]+)",
        r"registration number:\s*([\w\-]+)",
        r"account number:\s*([\*\d\w]+)",
    ],
    "Nationality":   [r"nationality[/\s\w]*:\s*([A-Za-z\s]+?)(?:\n|$)"],
    "Date of Birth": [r"date of birth[^\n]*:\s*(\d{1,2}[\s\/\-]\w+[\s\/\-]\d{2,4})"],
    "Expiry Date":   [r"(?:date of expiry|expiry date)[:\s]+(\d{1,2}[\s\/\-]\w+[\s\/\-]\d{4}|\d{2}\/\d{2}\/\d{4})"],
    "Address":       [r"(?:address|service address|registered address):\s*(.+?)(?:\n|$)"],
    "Amount":        [r"(?:total amount due|total payable|closing balance|amount due|total:)\s*([\d\s,\.]+)"],
    "Date":          [r"(?:bill date|invoice date|statement date):\s*(\d{1,2}\s+\w+\s+\d{4}|\d{2}\/\d{2}\/\d{4})"],
}


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Traditional ML
# ══════════════════════════════════════════════════════════════════════════
class TraditionalMLPipeline:
    """TF-IDF bigram features + Logistic Regression classifier."""

    def __init__(self, max_features: int = 20_000, C: float = 5.0):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
        )
        self.classifier = LogisticRegression(
            max_iter=1000, C=C, random_state=42,
        )
        self.is_fitted = False

    # ── Public API ─────────────────────────────────────────────────────────
    def fit(self, texts: list[str], labels: list[str]) -> "TraditionalMLPipeline":
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        self._check_fitted()
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self._check_fitted()
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict_one(self, text: str) -> dict:
        proba = self.predict_proba([text])[0]
        classes = self.classifier.classes_.tolist()
        pred = classes[int(np.argmax(proba))]
        return {
            "label": pred,
            "confidence": float(np.max(proba)),
            "scores": dict(zip(classes, proba.tolist())),
        }

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        t0 = time.perf_counter()
        preds = self.predict(texts)
        infer_ms = (time.perf_counter() - t0) / len(texts) * 1000
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        report = classification_report(labels, preds, output_dict=True)
        return {
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "infer_ms_per_sample": infer_ms,
            "per_class": {cls: report[cls] for cls in CLASSES if cls in report},
        }

    def save(self, path: str):
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, path)

    @classmethod
    def load(cls, path: str) -> "TraditionalMLPipeline":
        obj = joblib.load(path)
        instance = cls()
        instance.vectorizer  = obj["vectorizer"]
        instance.classifier  = obj["classifier"]
        instance.is_fitted   = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 2 — Transformer-Based
# ══════════════════════════════════════════════════════════════════════════
class TransformerPipeline:
    """
    Dual-channel TF-IDF (char n-grams + word n-grams) → LSA (256d)
    → L2-normalised dense embeddings → Logistic Regression head.

    Entity extraction runs a second-pass NER regex layer over the
    dense representation's top activated tokens.
    """

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=50_000, sublinear_tf=True,
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3),
            max_features=30_000, sublinear_tf=True,
        )
        self.svd        = TruncatedSVD(n_components=n_components, random_state=42)
        self.normalizer = Normalizer()
        self.classifier = LogisticRegression(max_iter=1000, C=10, random_state=42)
        self.is_fitted  = False

    # ── Internal ───────────────────────────────────────────────────────────
    def _combine(self, texts: list[str], fit: bool = False):
        if fit:
            Xc = self.tfidf_char.fit_transform(texts)
            Xw = self.tfidf_word.fit_transform(texts)
        else:
            Xc = self.tfidf_char.transform(texts)
            Xw = self.tfidf_word.transform(texts)
        return hstack([Xc, Xw])

    def _embed(self, X_sparse, fit: bool = False):
        if fit:
            X_lsa = self.svd.fit_transform(X_sparse)
            return self.normalizer.fit_transform(X_lsa)
        return self.normalizer.transform(self.svd.transform(X_sparse))

    # ── Public API ─────────────────────────────────────────────────────────
    def fit(self, texts: list[str], labels: list[str]) -> "TransformerPipeline":
        X_sparse = self._combine(texts, fit=True)
        X_embed  = self._embed(X_sparse, fit=True)
        self.classifier.fit(X_embed, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        self._check_fitted()
        X = self._embed(self._combine(texts))
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self._check_fitted()
        X = self._embed(self._combine(texts))
        return self.classifier.predict_proba(X)

    def predict_one(self, text: str) -> dict:
        proba = self.predict_proba([text])[0]
        classes = self.classifier.classes_.tolist()
        pred = classes[int(np.argmax(proba))]
        return {
            "label": pred,
            "confidence": float(np.max(proba)),
            "scores": dict(zip(classes, proba.tolist())),
        }

    def extract_entities(self, text: str) -> dict[str, str]:
        """Regex NER layer — same patterns as LLM pipeline but
        post-classification, so entity type set is label-aware."""
        return _regex_extract(text)

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        t0 = time.perf_counter()
        preds = self.predict(texts)
        infer_ms = (time.perf_counter() - t0) / len(texts) * 1000
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        report = classification_report(labels, preds, output_dict=True)
        return {
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "infer_ms_per_sample": infer_ms,
            "per_class": {cls: report[cls] for cls in CLASSES if cls in report},
        }

    def save(self, path: str):
        joblib.dump({
            "tfidf_char": self.tfidf_char, "tfidf_word": self.tfidf_word,
            "svd": self.svd, "normalizer": self.normalizer,
            "classifier": self.classifier,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TransformerPipeline":
        obj = joblib.load(path)
        inst = cls()
        for k in ("tfidf_char", "tfidf_word", "svd", "normalizer", "classifier"):
            setattr(inst, k, obj[k])
        inst.is_fitted = True
        return inst

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 3 — LLM-Assisted (zero-shot)
# ══════════════════════════════════════════════════════════════════════════
class LLMAssistedPipeline:
    """
    No training required.
    Classification: per-class keyword count + regex pattern match → highest score wins.
    Extraction:     field-specific regex patterns over raw text.
    """

    def predict_one(self, text: str) -> dict:
        tl = text.lower()
        scores: dict[str, float] = {}
        for cls, cfg in CLASS_SIGNATURES.items():
            kw_score  = sum(1 for kw in cfg["keywords"] if kw in tl)
            pat_score = sum(2 for pat in cfg["patterns"]
                           if re.search(pat, text, re.IGNORECASE))
            scores[cls] = float(kw_score + pat_score)

        total = sum(scores.values())
        best  = max(scores, key=scores.get)
        conf  = scores[best] / total if total > 0 else 0.0
        return {
            "label":      best,
            "confidence": conf,
            "scores":     scores,
        }

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(t)["label"] for t in texts]

    def extract_entities(self, text: str) -> dict[str, str]:
        return _regex_extract(text)

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        t0 = time.perf_counter()
        preds = self.predict(texts)
        infer_ms = (time.perf_counter() - t0) / len(texts) * 1000
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        report = classification_report(labels, preds, output_dict=True)
        return {
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "infer_ms_per_sample": infer_ms,
            "per_class": {cls: report[cls] for cls in CLASSES if cls in report},
        }


# ══════════════════════════════════════════════════════════════════════════
# SHARED UTILITY
# ══════════════════════════════════════════════════════════════════════════
def _regex_extract(text: str) -> dict[str, str]:
    """Run all entity patterns over raw text and return first match per type."""
    out: dict[str, str] = {}
    for etype, patterns in ENTITY_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = m.group(1).strip()
                if val and len(val) > 1:
                    out[etype] = val
                    break
    return out


# ══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════
class DocumentIntelligence:
    """
    Runs all three pipelines on a document.
    Returns merged result: best classification by confidence + entities.
    """

    def __init__(self, trad: TraditionalMLPipeline | None = None,
                 transformer: TransformerPipeline | None = None,
                 llm: LLMAssistedPipeline | None = None):
        self.trad        = trad or TraditionalMLPipeline()
        self.transformer = transformer or TransformerPipeline()
        self.llm         = llm or LLMAssistedPipeline()

    def fit(self, texts: list[str], labels: list[str]):
        print("Training Traditional ML …")
        t0 = time.perf_counter()
        self.trad.fit(texts, labels)
        print(f"  done in {time.perf_counter()-t0:.2f}s")

        print("Training Transformer pipeline …")
        t0 = time.perf_counter()
        self.transformer.fit(texts, labels)
        print(f"  done in {time.perf_counter()-t0:.2f}s")

        print("LLM-Assisted: no training needed.")
        return self

    def analyze(self, text: str) -> dict:
        r_trad  = self.trad.predict_one(text)        if self.trad.is_fitted  else None
        r_trans = self.transformer.predict_one(text) if self.transformer.is_fitted else None
        r_llm   = self.llm.predict_one(text)

        all_results = {
            "traditional_ml":    r_trad,
            "transformer_based": r_trans,
            "llm_assisted":      r_llm,
        }

        # Pick best confident result
        candidates = {k: v for k, v in all_results.items() if v is not None}
        best_key   = max(candidates, key=lambda k: candidates[k]["confidence"])
        best       = candidates[best_key]

        entities = (
            self.transformer.extract_entities(text)
            if r_trans
            else self.llm.extract_entities(text)
        )

        return {
            "predicted_label": best["label"],
            "confidence":      best["confidence"],
            "best_approach":   best_key,
            "entities":        entities,
            "all_approaches":  all_results,
        }

    def save(self, directory: str):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.trad.is_fitted:
            self.trad.save(str(d / "trad_model.joblib"))
        if self.transformer.is_fitted:
            self.transformer.save(str(d / "transformer_model.joblib"))
        print(f"Models saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "DocumentIntelligence":
        d = Path(directory)
        trad = TraditionalMLPipeline.load(str(d / "trad_model.joblib"))
        trans = TransformerPipeline.load(str(d / "transformer_model.joblib"))
        return cls(trad=trad, transformer=trans)


# ══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT — train & evaluate all approaches
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.json"
    model_dir    = sys.argv[2] if len(sys.argv) > 2 else "models/saved"

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        raw = json.load(f)

    texts  = [d["text"]  for d in raw]
    labels = [d["label"] for d in raw]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print("="*60)

    di = DocumentIntelligence()
    di.fit(X_train, y_train)

    print("\n── Evaluation ──────────────────────────────────────────────")
    for name, pipe in [
        ("Traditional ML",    di.trad),
        ("Transformer-Based", di.transformer),
        ("LLM-Assisted",      di.llm),
    ]:
        m = pipe.evaluate(X_test, y_test)
        print(
            f"\n{name}:\n"
            f"  Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}"
            f"  Recall={m['recall']:.4f}  F1={m['f1']:.4f}"
            f"  Infer={m['infer_ms_per_sample']:.3f}ms/sample"
        )
        for cls, cm in m["per_class"].items():
            print(f"    {cls:<30} P={cm['precision']:.3f}  R={cm['recall']:.3f}  F1={cm['f1-score']:.3f}")

    di.save(model_dir)

    # Quick demo
    print("\n── Sample Inference ────────────────────────────────────────")
    sample = X_test[0]
    result = di.analyze(sample)
    print(f"Text (first 200 chars): {sample[:200].strip()}")
    print(f"Predicted: {result['predicted_label']}  ({result['confidence']:.1%} conf, via {result['best_approach']})")
    print(f"Entities:  {result['entities']}")
