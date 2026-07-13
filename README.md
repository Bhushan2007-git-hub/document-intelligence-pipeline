# ML Document Intelligence — Classification & Entity Extraction

**Invent SoftLabs Internship Project · Bhushan (25BAI1249) · VIT Chennai**

End-to-end ML pipeline that classifies uploaded documents into 6 categories
and extracts key entities using three distinct approaches.

---

## Project Structure

```
project/
├── demo/
│   └── appdemo.html              # Interactive demo — open directly in browser
├── src/
│   ├── data/
│   │   └── generate_data.py      # Synthetic dataset generator (6 classes × N samples)
│   ├── models/
│   │   └── pipeline.py           # All 3 ML approaches + orchestrator
│   └── backend/
│       └── main.py               # FastAPI REST API
├── scripts/
│   ├── train.py                  # One-command: generate → train → evaluate → save
│   └── evaluate.py               # Load saved models, print metrics + confusion matrix
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
py -3.13 -m pip install -r requirements.txt

# 2. Generate data, train all models, evaluate
py -3.13 scripts/train.py

# 3. Start the API
py -3.13 -m uvicorn src.backend.main:app --reload --port 8000
```

Then open `demo/appdemo.html` in your browser.

---

## Demo

Open `demo/appdemo.html` directly in any browser — no server needed for the charts, metrics, and comparison pages.

For the **Live Demo** tab to work (real ML classification), start the API first with the command above. The demo has two modes:
- **Sample Documents** — pick from 6 pre-loaded document types
- **Paste Your Own Text** — paste any document text and run it through the real pipeline

---

## Document Classes

| Class | Key Entities |
|---|---|
| Passport | Name, Nationality, DOB, Expiry, MRZ |
| National ID | Name, ID Number, Blood Type, Address, Expiry |
| Commercial Registration | Company, Reg. No., Owner, Capital, Activity |
| Bank Statement | Account Holder, IBAN, Account No., Closing Balance |
| Utility Bill | Customer, Meter No., Units Consumed, Amount Due |
| Invoice | Vendor, Client, Invoice No., Total Amount |

---

## Approaches

### Traditional ML — `TraditionalMLPipeline`
- TF-IDF (bigrams, 20k features, sublinear TF) + Logistic Regression
- Train: ~0.3s · Infer: ~0.15ms/sample · RAM: 45 MB
- Best for: high-volume pipelines, resource-constrained environments

### Transformer-Based — `TransformerPipeline`
- Dual-channel TF-IDF (char 2–4grams + word 1–3grams) → LSA (256d) → L2-norm → LogReg head
- Train: ~13s · Infer: ~0.001ms/sample · RAM: 890 MB
- Best for: maximum accuracy, noisy/multilingual documents, NER extraction

### LLM-Assisted — `LLMAssistedPipeline`
- Per-class keyword scoring + regex pattern matching (zero training)
- Train: 0s · Infer: ~0.5ms/sample · RAM: 28 MB
- Best for: rapid prototyping, new document types, interpretable decisions

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify` | Classify document text → label + confidence |
| `POST` | `/extract` | Extract entities from text |
| `POST` | `/analyze` | Full: classify + extract + all approach scores |
| `POST` | `/analyze/file` | Upload .txt file and analyze |
| `GET` | `/classes` | List supported document classes |
| `GET` | `/health` | Liveness check |

API docs available at: `http://localhost:8000/docs`

### Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "PASSPORT\nRepublic of India\nPassport No.: Z8234567\nNationality: Indian\n..."}'
```

### Example response

```json
{
  "label": "Passport",
  "confidence": 0.97,
  "best_approach": "transformer_based",
  "entities": {
    "Name": "Bhushan Sharma",
    "Nationality": "Indian",
    "Expiry Date": "09 Jan 2034"
  },
  "all_approaches": {
    "traditional_ml":    {"label": "Passport", "confidence": 0.94},
    "transformer_based": {"label": "Passport", "confidence": 0.97},
    "llm_assisted":      {"label": "Passport", "confidence": 0.86}
  },
  "inference_ms": 1.24
}
```

---

## Results Summary (240 test samples, 40/class)

| Approach | Accuracy | F1-Score | Train Time | Infer/sample |
|---|---|---|---|---|
| Traditional ML | 96.67% | 96.65% | 0.26s | 0.146ms |
| Transformer-Based | **99.17%** | **99.17%** | 13.26s | 0.001ms |
| LLM-Assisted | 98.33% | 98.33% | 0s | 0.482ms |

Entity extraction (partial match recall):
- Transformer-Based NER: **74%**
- LLM-Assisted Regex: 52%

---

## Production Notes

- Recommended: hybrid pipeline — Traditional ML fast path (conf > 0.95) → Transformer NER → human review
- Add OCR pre-processing (PaddleOCR / Tesseract) for image/scanned PDF inputs
- Encrypt extracted PII fields at rest (AES-256)
- Monitor prediction distribution drift weekly (PSI > 0.2 → retrain)
