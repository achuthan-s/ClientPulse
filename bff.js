// BFF (Backend-for-Frontend) — Express layer between React and FastAPI.
// Responsibility: aggregate/shape data for the UI so the frontend
// never talks to FastAPI directly. Keeps the API contract stable even
// if the Python backend changes internally.

const express = require("express");
const axios = require("axios");
const app = express();
app.use(express.json());

const API_BASE = process.env.CLIENTPULSE_API || "http://localhost:8000";

// ---------------------------------------------------------------------------
// HELPER: forward errors from FastAPI as structured JSON (not HTML stacks)
// ---------------------------------------------------------------------------
function fwdError(res, e) {
  const status = e.response?.status || 500;
  const detail = e.response?.data?.detail || e.toString();
  res.status(status).json({ error: detail });
}

// ---------------------------------------------------------------------------
// EXISTING: /api/dashboard
// Returns flags + a summary count by risk_level.
// Unchanged shape — Dashboard.jsx already consumes this.
// ---------------------------------------------------------------------------
app.get("/api/dashboard", async (req, res) => {
  try {
    const flags = (await axios.get(`${API_BASE}/flags`)).data;
    const summary = flags.reduce((acc, f) => {
      acc[f.risk_level] = (acc[f.risk_level] || 0) + 1;
      return acc;
    }, {});
    res.json({ flags, summary });
  } catch (e) {
    fwdError(res, e);
  }
});

// ---------------------------------------------------------------------------
// NEW: /api/accounts
// Returns all accounts enriched with trend metrics + sparkline series.
// Used by:
//   - Dashboard account list (Step 1) for risk badges and sparklines
//   - Copilot dropdown (Step 4) to populate the <select>
// ---------------------------------------------------------------------------
app.get("/api/accounts", async (req, res) => {
  try {
    const data = (await axios.get(`${API_BASE}/accounts`)).data;
    res.json(data);
  } catch (e) {
    fwdError(res, e);
  }
});

// ---------------------------------------------------------------------------
// UPGRADED: /api/account/:id
// Now returns date-stamped series, full message list with reason/topics,
// flagged evidence, and a decline_start_index marker.
// ---------------------------------------------------------------------------
app.get("/api/account/:id", async (req, res) => {
  try {
    const data = (await axios.get(`${API_BASE}/account/${req.params.id}`)).data;
    res.json(data);
  } catch (e) {
    fwdError(res, e);
  }
});

// ---------------------------------------------------------------------------
// NEW: /api/account/:id/history?before=YYYY-MM-DD
// Time-machine endpoint: recomputes trend as of a given date.
// Used by the time-machine slider (Step 3).
// ---------------------------------------------------------------------------
app.get("/api/account/:id/history", async (req, res) => {
  try {
    const params = req.query.before ? `?before=${req.query.before}` : "";
    const data = (
      await axios.get(`${API_BASE}/account/${req.params.id}/history${params}`)
    ).data;
    res.json(data);
  } catch (e) {
    fwdError(res, e);
  }
});

// ---------------------------------------------------------------------------
// NEW: /api/eval
// Returns Phase-6 evaluation results: accuracy, timing, false positives.
// Used by the Evaluation Summary card (Step 5).
// ---------------------------------------------------------------------------
app.get("/api/eval", async (req, res) => {
  try {
    const data = (await axios.get(`${API_BASE}/eval`)).data;
    res.json(data);
  } catch (e) {
    fwdError(res, e);
  }
});

// ---------------------------------------------------------------------------
// NEW: /api/account/:id/draft
// Generates a check-in email draft grounded in the account's evidence.
// Used by the "Suggested Action" button (Step 6).
// ---------------------------------------------------------------------------
app.post("/api/account/:id/draft", async (req, res) => {
  try {
    const data = (
      await axios.post(`${API_BASE}/account/${req.params.id}/draft`, {})
    ).data;
    res.json(data);
  } catch (e) {
    fwdError(res, e);
  }
});

// ---------------------------------------------------------------------------
// EXISTING: /api/ask — pass-through to FastAPI /ask
// ---------------------------------------------------------------------------
app.post("/api/ask", async (req, res) => {
  try {
    const body = {
      account_id: req.body.account_id,
      question:   req.body.question,
    };
    const result = (await axios.post(`${API_BASE}/ask`, body)).data;
    res.json(result);
  } catch (e) {
    fwdError(res, e);
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`BFF listening on ${port}`));
