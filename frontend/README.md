# ClientPulse Frontend (React + Vite)

This is a minimal React frontend scaffold for the ClientPulse project.

Run locally:

1. cd frontend
2. npm install
3. npm run dev

The frontend expects a BFF at `/api` (see `../bff.js`) which forwards
requests to the FastAPI backend running at `http://localhost:8000`.

BFF:

1. Install dependencies: `npm install express axios`
2. Run: `node bff.js`

FastAPI backend:

Run `python3 -m uvicorn backend:app --reload` from the project root.
