# DDM501 Lab 1 — Credit Default Risk Scoring API

**Student:** Phạm Văn Duy Khánh  
**Student ID:** 25MS13313

FastAPI service that loads a trained credit-default model and exposes `/health`, `/predict`, `/predict/batch`, and `/model/info`. Input validation is enforced with Pydantic and the runtime image uses a non-root user.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py
pytest tests -v --cov=app --cov-report=term-missing
uvicorn app.main:app --reload
```

Swagger UI is available at `http://localhost:8000/docs`.

## Run with Docker

Generate the model once, then start the container:

```bash
python scripts/train_model.py
docker compose up --build
```

The model is mounted read-only from `./models`; it is deliberately not baked into the image.

## Decision policy

| Probability | Risk band | Decision |
|---|---|---|
| `< 0.30` | LOW | APPROVE |
| `0.30–0.59` | MEDIUM | REVIEW |
| `>= 0.60` | HIGH | DECLINE |
