from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
import os
import sqlite3
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

import risk_flags
import agent
import rag_copilot
from trend_tracker import (
    init_db,
    compute_account_trend,
    classify_risk,
    get_all_account_ids,
    get_sentiment_series,
    rolling_average,
)
from hybrid_search_eval import GROUND_TRUTH

app = FastAPI(title="ClientPulse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    account_id: str
    question: str


# ---------------------------------------------------------------------------
# Startup: load shared resources once so each request doesn't reload them.
# The message JSON is small (60 msgs) so keeping it in memory is fine.
# ---------------------------------------------------------------------------
MESSAGES = agent.load_messages()
TEXTS = [m["text"] for m in MESSAGES]

# Index messages by account_id for O(1) per-account lookup
MESSAGES_BY_ACCOUNT: dict = {}
for msg in MESSAGES:
    MESSAGES_BY_ACCOUNT.setdefault(msg["account_id"], []).append(msg)
for acct in MESSAGES_BY_ACCOUNT:
    MESSAGES_BY_ACCOUNT[acct].sort(key=lambda m: m["date"])

try:
    from hybrid_search_eval import BM25 as BM25Cls
    BM25 = BM25Cls(TEXTS)
except Exception:
    BM25 = None

try:
    import chromadb
    from chromadb.utils import embedding_functions
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    COLLECTION = rag_copilot.safe_get_collection(
        chroma_client, "client_messages", sentence_transformer_ef
    )
except Exception:
    COLLECTION = None

DB_CONN = init_db()


# ---------------------------------------------------------------------------
# HELPER: Build the full date-stamped series for one account.
# trend_tracker.compute_account_trend() returns smoothed_series as a plain
# float list with no dates. We re-join with the raw rows here so the
# frontend can plot a proper time-axis.
# ---------------------------------------------------------------------------
def _build_series(conn: sqlite3.Connection, account_id: str) -> list:
    """
    Returns a list of {date, raw, smoothed} dicts, one per message.
    The smoothed value uses the same window=4 rolling average as compute_account_trend.
    """
    rows = get_sentiment_series(conn, account_id)   # [(date, sentiment), ...]
    if not rows:
        return []
    sentiments = [r[1] for r in rows]
    smoothed   = rolling_average(sentiments, window=4)
    return [
        {"date": rows[i][0][:10], "raw": round(rows[i][1], 3), "smoothed": round(smoothed[i], 3)}
        for i in range(len(rows))
    ]


def _enrich_flag(flag: dict) -> dict:
    """
    Adds the date-stamped sparkline series to a flag record so the
    dashboard can render a mini chart per account row.
    """
    flag["series"] = _build_series(DB_CONN, flag["account_id"])
    return flag


# ---------------------------------------------------------------------------
# EXISTING ENDPOINT — unchanged behaviour, kept for backwards compat
# ---------------------------------------------------------------------------
@app.get("/flags")
def get_flags():
    """
    Returns all risk flags.  list_flags() in risk_flags.py queries the
    risk_flags table which was populated by the Phase-9 agent pipeline.
    """
    return risk_flags.list_flags(DB_CONN)


# ---------------------------------------------------------------------------
# NEW: /accounts — all accounts with sparkline series + risk metadata.
# Used by the dashboard list (Step 1) and the Copilot dropdown (Step 4).
# ---------------------------------------------------------------------------
@app.get("/accounts")
def get_accounts():
    """
    Returns every account the system knows about (from sentiment_history),
    enriched with:
      - trend metrics (slope, drift, baseline, recent_avg)
      - risk_level from the flags table (if flagged) or from classify_risk
      - date-stamped series for the sparkline chart
    """
    account_ids = get_all_account_ids(DB_CONN)
    result = []

    # Load flag lookup so we can merge the stored explanation
    flag_rows = risk_flags.list_flags(DB_CONN)
    flag_map = {f["account_id"]: f for f in flag_rows}

    for acct in sorted(account_ids):
        trend = compute_account_trend(DB_CONN, acct)
        if not trend:
            continue

        risk = flag_map.get(acct, {}).get("risk_level") or classify_risk(trend)
        explanation = flag_map.get(acct, {}).get("explanation", "")

        result.append({
            "account_id":  acct,
            "risk_level":  risk,
            "explanation": explanation,
            "baseline":    trend["baseline"],
            "recent_avg":  trend["recent_avg"],
            "slope":       trend["slope"],
            "drift":       trend["drift"],
            "first_date":  trend["first_date"],
            "last_date":   trend["last_date"],
            "n_messages":  trend["n_messages"],
            "series":      _build_series(DB_CONN, acct),
        })

    return result


# ---------------------------------------------------------------------------
# UPGRADED: /account/{account_id} — now returns date-stamped series,
# full evidence with reason/topics, and baseline marker.
# ---------------------------------------------------------------------------
@app.get("/account/{account_id}")
def get_account(account_id: str):
    """
    Upgraded from the original: trend now includes a date-stamped `series`
    list (not just smoothed floats), and evidence messages include the
    `reason` and `topics` fields from the JSON so the frontend can
    highlight key phrases and show the flag reason inline.
    """
    trend = compute_account_trend(DB_CONN, account_id)
    if not trend:
        raise HTTPException(
            status_code=404,
            detail="Account not found or no sentiment history"
        )

    # Classify risk and add the date-stamped series
    trend["risk"]   = classify_risk(trend)
    trend["series"] = _build_series(DB_CONN, account_id)

    # Find the index where the smoothed sentiment first crosses below the baseline —
    # this is the "decline began" marker we'll show on the chart.
    baseline = trend["baseline"]
    decline_start_index = None
    for i, pt in enumerate(trend["series"]):
        if pt["smoothed"] < baseline and i > 0:
            decline_start_index = i
            break
    trend["decline_start_index"] = decline_start_index

    # Full message history for this account, sorted chronologically.
    # We use MESSAGES_BY_ACCOUNT (loaded at startup) so we get the
    # `reason` and `topics` fields that live in the JSON but not the DB.
    all_msgs = MESSAGES_BY_ACCOUNT.get(account_id, [])

    # Flagged evidence: messages from the risk_flags table (enriched)
    flag = risk_flags.get_flag(DB_CONN, account_id)
    evidence = flag["evidence"] if flag else []

    return {
        "trend":           trend,
        "all_messages":    all_msgs,       # full history for timeline
        "flagged_evidence": evidence,       # stored evidence snippets
        "flag":            flag,
    }


# ---------------------------------------------------------------------------
# NEW: /account/{account_id}/history — time-machine endpoint.
# Recomputes the trend as if the system only knew about messages up to
# `before` (inclusive). Used by the time-machine slider (Step 3).
# ---------------------------------------------------------------------------
@app.get("/account/{account_id}/history")
def get_account_history(
    account_id: str,
    before: Optional[str] = Query(
        None,
        description="ISO-8601 date string (YYYY-MM-DD). Only include messages on or before this date."
    ),
):
    """
    Re-runs trend computation on a date-filtered subset of sentiment_history.
    Uses an in-memory SQLite DB so we don't touch the real DB at all —
    this is purely a read-only simulation of "what would the system have known?"
    """
    # Pull all rows for this account from the real DB
    all_rows = get_sentiment_series(DB_CONN, account_id)
    if not all_rows:
        raise HTTPException(status_code=404, detail="Account not found")

    # Filter to rows on or before the requested date
    if before:
        all_rows = [(d, s) for (d, s) in all_rows if d[:10] <= before]

    if not all_rows:
        return {"series": [], "trend": None, "risk": None}

    # Build the series response directly — no need for a full DB roundtrip
    sentiments = [r[1] for r in all_rows]
    smoothed   = rolling_average(sentiments, window=4)
    series     = [
        {"date": all_rows[i][0][:10], "raw": round(all_rows[i][1], 3), "smoothed": round(smoothed[i], 3)}
        for i in range(len(all_rows))
    ]

    # Recompute trend metrics inline (mirrors compute_account_trend logic)
    half        = max(1, len(sentiments) // 2)
    baseline    = sum(sentiments[:half]) / half
    recent_avg  = smoothed[-1]
    drift       = recent_avg - baseline

    from trend_tracker import compute_slope
    slope = compute_slope(sentiments)

    trend_snapshot = {
        "account_id": account_id,
        "baseline":   round(baseline, 3),
        "recent_avg": round(recent_avg, 3),
        "slope":      round(slope, 4),
        "drift":      round(drift, 3),
        "n_messages": len(sentiments),
        "first_date": all_rows[0][0][:10],
        "last_date":  all_rows[-1][0][:10],
    }
    risk = classify_risk(trend_snapshot)

    # Annotate when the decline starts (first point where smoothed < baseline)
    decline_start_index = None
    for i, pt in enumerate(series):
        if pt["smoothed"] < baseline and i > 0:
            decline_start_index = i
            break

    # Find the messages visible at this point in time (for the frontend
    # to show which evidence cards would have been visible)
    visible_messages = [
        m for m in MESSAGES_BY_ACCOUNT.get(account_id, [])
        if (not before or m["date"][:10] <= before)
    ]

    return {
        "series":               series,
        "trend":                trend_snapshot,
        "risk":                 risk,
        "decline_start_index":  decline_start_index,
        "visible_messages":     visible_messages,
    }


# ---------------------------------------------------------------------------
# NEW: /eval — pre-computed Phase-6 evaluation results.
# Static data: we run it once at startup so the endpoint is instant.
# The frontend uses this to show the Evaluation Summary card (Step 5).
# ---------------------------------------------------------------------------
def _run_eval_once() -> dict:
    """
    Re-runs the evaluation harness (timing + accuracy) in-memory.
    Called once at startup and cached in EVAL_RESULT.
    """
    import sqlite3 as _sqlite3
    from trend_tracker import populate_db, compute_account_trend, classify_risk

    # --- Classification accuracy ---
    should_flag     = set(GROUND_TRUTH["should_be_flagged"])
    should_not_flag = set(GROUND_TRUTH["should_not_be_flagged"])
    all_test_accts  = should_flag | should_not_flag

    classification_results = []
    correct = 0
    false_positives = 0
    false_negatives = 0

    for acct in sorted(all_test_accts):
        trend   = compute_account_trend(DB_CONN, acct)
        risk    = classify_risk(trend) if trend else "STABLE"
        flagged = "HIGH RISK" in risk or "WATCH" in risk
        expected = acct in should_flag

        is_correct   = flagged == expected
        is_fp        = flagged and not expected   # stable account falsely flagged
        is_fn        = not flagged and expected   # at-risk account missed

        correct        += 1 if is_correct else 0
        false_positives += 1 if is_fp else 0
        false_negatives += 1 if is_fn else 0

        classification_results.append({
            "account_id": acct,
            "flagged":    flagged,
            "expected":   expected,
            "correct":    is_correct,
            "risk_level": risk,
        })

    # --- Timing: how early does TechCorp get flagged vs designed trigger? ---
    techcorp_msgs = sorted(
        [m for m in MESSAGES if m["account_id"] == "acct_techcorp"],
        key=lambda m: m["date"],
    )
    designed_trigger_index = GROUND_TRUTH["techcorp_decline_starts_at_index"]
    flagged_at_index = None
    flagged_at_date  = None
    msgs_before_trigger = None

    for i in range(2, len(techcorp_msgs) + 1):
        subset = techcorp_msgs[:i]
        conn2  = _sqlite3.connect(":memory:")
        conn2.execute("""
            CREATE TABLE sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                account_id TEXT, date TEXT,
                sentiment REAL,
                scope_dispute_signal INTEGER,
                channel TEXT
            )
        """)
        conn2.commit()
        populate_db(conn2, subset)
        trend2   = compute_account_trend(conn2, "acct_techcorp")
        risk2    = classify_risk(trend2) if trend2 else "STABLE"
        flagged2 = "HIGH RISK" in risk2 or "WATCH" in risk2
        conn2.close()

        if flagged2:
            flagged_at_index    = i - 1  # 0-indexed
            flagged_at_date     = subset[-1]["date"][:10]
            # Positive = early (flagged BEFORE designed trigger)
            msgs_before_trigger = designed_trigger_index - flagged_at_index
            break

    return {
        "accuracy":              correct / len(all_test_accts),
        "correct":               correct,
        "total":                 len(all_test_accts),
        "false_positives":       false_positives,
        "false_negatives":       false_negatives,
        "classification":        classification_results,
        "timing": {
            "account":                    "acct_techcorp",
            "designed_trigger_index":     designed_trigger_index,
            "flagged_at_index":           flagged_at_index,
            "flagged_at_date":            flagged_at_date,
            # Positive number = system flagged N messages BEFORE designed trigger
            "msgs_earlier_than_designed": msgs_before_trigger,
        },
    }


# Run once at startup; cached for the lifetime of the process
EVAL_RESULT = _run_eval_once()


@app.get("/eval")
def get_eval():
    """
    Returns the Phase-6 evaluation results: classification accuracy,
    false positive / negative counts, and the early-detection timing
    for TechCorp vs its designed decline trigger point.
    """
    return EVAL_RESULT


# ---------------------------------------------------------------------------
# NEW: /account/{account_id}/draft — generate a check-in message draft.
# Calls the LLM with a targeted prompt grounded in the flag evidence.
# ---------------------------------------------------------------------------
class DraftRequest(BaseModel):
    account_id: str


@app.post("/account/{account_id}/draft")
def draft_checkin(account_id: str):
    """
    Uses the RAG copilot's LLM call to draft a short, warm check-in
    message to the account owner, grounded in the stored flag evidence.
    The draft is deliberately kept brief (3–4 sentences) so the AM can
    personalise it before sending.
    """
    flag = risk_flags.get_flag(DB_CONN, account_id)
    if not flag:
        raise HTTPException(
            status_code=404,
            detail=f"No risk flag found for {account_id}"
        )

    trend = compute_account_trend(DB_CONN, account_id)
    trend_summary = ""
    if trend:
        trend_summary = (
            f"Trend slope: {trend['slope']:+.4f}, "
            f"drift from baseline: {trend['drift']:+.3f}, "
            f"period: {trend['first_date']} → {trend['last_date']}"
        )

    # Format stored evidence so the LLM can reference specific messages
    evidence_lines = "\n".join(
        f"- [{e['date'][:10]} | {e['channel']}]: {e['text_snippet']}"
        for e in flag["evidence"][:5]
    )

    system_prompt = (
        "You are a helpful assistant drafting a brief, professional, empathetic "
        "check-in email for an account manager to send to a client. "
        "The email should be warm but not alarming, acknowledge the specific "
        "concerns raised in the evidence, and invite the client to a call. "
        "Keep it to 3–4 sentences. Do NOT mention internal metrics or system scores."
    )

    user_prompt = (
        f"Account: {account_id}\n"
        f"Risk flags: {flag['explanation']}\n"
        f"Trend: {trend_summary}\n\n"
        f"Evidence messages:\n{evidence_lines}\n\n"
        "Draft a short, warm check-in email from the account manager to the client."
    )

    draft = rag_copilot.call_llm(system_prompt, user_prompt)
    return {
        "account_id": account_id,
        "draft":      draft,
        "based_on":   flag["evidence"][:5],
    }


# ---------------------------------------------------------------------------
# EXISTING ENDPOINT — unchanged
# ---------------------------------------------------------------------------
@app.post("/ask")
def ask(request: AskRequest):
    """Delegates to rag_copilot.ask_copilot for grounded RAG answers."""
    result = rag_copilot.ask_copilot(
        request.question,
        request.account_id,
        MESSAGES,
        BM25,
        COLLECTION,
        DB_CONN,
    )
    return result
