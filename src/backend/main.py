"""
backend/main.py
FastAPI REST API for the Document Intelligence pipeline.

Endpoints
---------
POST /classify          – classify plain text, returns label + confidence
POST /extract           – extract entities from plain text
POST /analyze           – full analysis: classify + extract + all approaches
POST /analyze/file      – upload a .txt file and run full analysis
GET  /classes           – list supported document classes
GET  /health            – liveness check
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import pipeline (adjust path if running from project root) ─────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pipeline import (
    CLASSES,
    DocumentIntelligence,
    LLMAssistedPipeline,
    TraditionalMLPipeline,
    TransformerPipeline,
    _regex_extract,
)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Intelligence API",
    description=(
        "ML pipeline for automatic document classification & entity extraction. "
        "Supports Passport, National ID, Commercial Registration, Bank Statement, "
        "Utility Bill, and Invoice."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "models/saved")
_di: DocumentIntelligence | None = None


def get_di() -> DocumentIntelligence:
    global _di
    if _di is None:
        model_path = Path(MODEL_DIR)
        if (model_path / "trad_model.joblib").exists():
            print(f"Loading saved models from {MODEL_DIR} …")
            _di = DocumentIntelligence.load(MODEL_DIR)
        else:
            # Fall back: LLM-only (zero-shot, no training needed)
            print("No saved models found — running LLM-Assisted only (zero-shot).")
            _di = DocumentIntelligence(llm=LLMAssistedPipeline())
    return _di


# ── Request / Response schemas ─────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    approach: str = "best"   # "best" | "traditional" | "transformer" | "llm"


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    approach_used: str
    inference_ms: float
    all_scores: dict[str, float] | None = None


class ExtractResponse(BaseModel):
    entities: dict[str, str]
    inference_ms: float


class AnalyzeResponse(BaseModel):
    label: str
    confidence: float
    best_approach: str
    entities: dict[str, str]
    all_approaches: dict[str, Any]
    inference_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────
def _run_approach(di: DocumentIntelligence, text: str, approach: str) -> dict:
    if approach == "traditional":
        if not di.trad.is_fitted:
            raise HTTPException(422, "Traditional ML model not loaded.")
        return di.trad.predict_one(text)
    if approach == "transformer":
        if not di.transformer.is_fitted:
            raise HTTPException(422, "Transformer model not loaded.")
        return di.transformer.predict_one(text)
    if approach == "llm":
        return di.llm.predict_one(text)
    # "best" — run all available, pick highest confidence
    return di.analyze(text)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/classes")
def list_classes():
    return {"classes": CLASSES, "count": len(CLASSES)}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty.")
    di = get_di()
    t0 = time.perf_counter()
    if req.approach == "best":
        result = di.analyze(req.text)
        label      = result["predicted_label"]
        confidence = result["confidence"]
        approach   = result["best_approach"]
        scores     = result["all_approaches"].get(approach, {}).get("scores")
    else:
        result     = _run_approach(di, req.text, req.approach)
        label      = result["label"]
        confidence = result["confidence"]
        approach   = req.approach
        scores     = result.get("scores")
    return ClassifyResponse(
        label=label,
        confidence=confidence,
        approach_used=approach,
        inference_ms=(time.perf_counter() - t0) * 1000,
        all_scores=scores,
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty.")
    t0 = time.perf_counter()
    entities = _regex_extract(req.text)
    return ExtractResponse(
        entities=entities,
        inference_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty.")
    di = get_di()
    t0 = time.perf_counter()
    result = di.analyze(req.text)
    return AnalyzeResponse(
        label=result["predicted_label"],
        confidence=result["confidence"],
        best_approach=result["best_approach"],
        entities=result["entities"],
        all_approaches=result["all_approaches"],
        inference_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/analyze/file", response_model=AnalyzeResponse)
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".text")):
        raise HTTPException(400, "Only .txt files are supported.")
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    if not text.strip():
        raise HTTPException(400, "Uploaded file is empty.")
    di = get_di()
    t0 = time.perf_counter()
    result = di.analyze(text)
    return AnalyzeResponse(
        label=result["predicted_label"],
        confidence=result["confidence"],
        best_approach=result["best_approach"],
        entities=result["entities"],
        all_approaches=result["all_approaches"],
        inference_ms=(time.perf_counter() - t0) * 1000,
    )
