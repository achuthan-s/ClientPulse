Run frontend and BFF locally (development)

1. Backend (FastAPI)
   - From project root: python3 -m uvicorn backend:app --reload

2. BFF (Node)
   - cd bff directory (project root)
   - Install: npm install express axios
   - Run: node bff.js

3. Frontend (React + Vite)
   - cd frontend
   - npm install
   - npm run dev

Note: Vite dev server proxies `/api` to the BFF at http://localhost:3000.
