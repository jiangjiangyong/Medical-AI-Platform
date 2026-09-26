# Medical Imaging Health Platform

A web application for case intake, medical-image upload, structured analysis, physician review,
report publication, and follow-up management.

## Structure

- `backend/` — FastAPI application, persistence, domain services, inference adapters, and tests
- `frontend/` — Vue 3 and Vite client
- `backend/knowledge/` — small reference knowledge base used by the application

Runtime data, uploaded files, local databases, model files, logs, and build output are intentionally
excluded from the repository.

## Backend setup

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python backend/scripts/seed_demo.py
uvicorn app.main:app --app-dir backend --env-file .env --reload --port 8000
```

The application reads its runtime configuration from `.env`. Use an available inference adapter
and provide credentials through environment variables or an external credentials file.

## Frontend setup

```bash
cd frontend
npm ci
npm run dev
```

The development server proxies `/api` requests to the backend. Set `VITE_DEV_API_TARGET` in
`frontend/.env` when the backend runs at another address.

## Tests

```bash
PYTHONPATH=backend pytest -q backend/tests
```

Do not commit secrets, patient data, uploaded images, or local database files. This repository is a
development foundation; production deployment still requires deployment-specific access control,
storage, auditing, and compliance review.
