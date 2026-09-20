# RiskLens — Explainable AI Credit Risk Assessment & Loan Recommendation

An end-to-end credit risk system: an XGBoost model predicts default
probability, SHAP explains every prediction feature-by-feature, and a
Next.js frontend (3D scroll-driven landing page + live dashboard) presents
the recommendation in a way a non-technical reviewer can actually trust.

## Architecture

```
Applicant data → FastAPI /predict → XGBoost model → SHAP TreeExplainer
                                                            │
                                   risk score + recommendation + explanation
                                                            │
                                              Next.js dashboard (charts)
```

## Project structure

```
credit-risk-xai/
├── backend/
│   ├── preprocessing.py      # shared feature engineering (train + serve)
│   ├── schemas.py            # Pydantic request/response models
│   ├── train.py              # data generation, training, SHAP, evaluation
│   ├── main.py                # FastAPI app + /predict endpoint
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/                # generated: xgb_credit_model.json, metadata.json
│   └── data/                  # generated: synthetic_credit_data.csv
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           # landing page, mounts CanvasScroll
│   │   ├── globals.css
│   │   └── dashboard/page.tsx # applicant form + risk/SHAP dashboard
│   ├── components/
│   │   └── CanvasScroll.tsx   # R3F + GSAP ScrollTrigger 3D hero
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── next.config.mjs
│   ├── vercel.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Backend — local setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py                 # generates synthetic data + trains the model
uvicorn main:app --reload       # http://localhost:8000  (docs at /docs)
```

Try it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":34,"annual_income":78000,"loan_amount":15000,"credit_score":705,"employment_length_years":6,"debt_to_income_ratio":0.28,"num_open_credit_lines":5,"num_credit_inquiries_last_6m":1,"credit_utilization_rate":0.32,"past_defaults":0,"home_ownership":"MORTGAGE","loan_purpose":"debt_consolidation"}'
```

## Frontend — local setup

```bash
cd frontend
npm install
cp .env.example .env.local      # edit NEXT_PUBLIC_API_URL if not localhost:8000
npm run dev                     # http://localhost:3000
```

## Docker Compose (both services)

```bash
docker compose up --build
```

## Deploying

- **Backend** — any host that runs a Docker container (Render, Railway,
  Fly.io, an EC2 box) using `backend/Dockerfile`.
- **Frontend** — push to GitHub, import into Vercel, and set the project's
  **Root Directory** to `frontend`. Set `NEXT_PUBLIC_API_URL` in the
  Vercel project's environment variables to point at your deployed backend.

## Verified locally

Both services were actually run (not just written) while building this:
`train.py` trains and evaluates in under a second (ROC-AUC ≈ 0.72 on the
synthetic data), `main.py` was exercised through strong / average / risky
applicant profiles and a validation-error case via FastAPI's `TestClient`,
and the frontend passes `tsc --noEmit` and a full `next build` cleanly.

## A note on the data — read this before you present it

`train.py` generates **synthetic** credit data with a hand-tuned, plausible
relationship between features and default risk. There is no real financial
data anywhere in this repo. That's the right call for a class project, but
before this touches anything resembling a real lending decision you'd need
to: (1) replace `generate_synthetic_dataset()` in `preprocessing.py` with a
loader for a real, licensed/anonymized credit dataset, and (2) run fairness
and disparate-impact audits across protected classes — credit scoring is
one of the more heavily regulated applications of ML (ECOA in the US, for
instance), and that's arguably as important a part of the system as the
model itself. Worth a paragraph in your report even if it's out of scope
for the build.
