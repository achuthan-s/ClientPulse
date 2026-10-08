"""
Phase 9: Risk Flag Generation
--------------------------------
Creates a persistent `risk_flags` table and a simple pipeline that:
 - runs the Phase 8 agent scanner to produce candidate accounts and evidence
 - converts those into a canonical `flag` object (account, risk_level,
   evidence, explanation, timestamp)
 - persists flags into the existing SQLite DB (data/clientpulse.db)

This module provides a small CLI to generate flags and to list persisted
flags. Flags are idempotent per account (single row per account) using
`INSERT OR REPLACE` with a UNIQUE constraint on `account_id`.

Why this exists
---------------
The agent (Phase 8) identifies candidate accounts and the signals that
triggered them. Phase 9 converts that transient output into durable
records that other systems (dashboard, notification pipelines) can read.
Each flag stores the human-readable explanation and the raw evidence used
to make the decision so nothing is opaque.
"""

import json
import os
from datetime import datetime

from typing import List

from trend_tracker import init_db, DB_PATH
import agent


FLAGS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL UNIQUE,
    risk_level TEXT NOT NULL,
    explanation TEXT,
    evidence_json TEXT,
    created_at TEXT NOT NULL
)
"""


def ensure_flags_table(conn):
    cur = conn.cursor()
    cur.execute(FLAGS_TABLE_SCHEMA)
    conn.commit()


def make_flag_record(item: dict) -> dict:
    """Convert a scanner item into a serializable flag record."""
    explanation = ", ".join(item.get("reasons", []))
    evidence = []
    for m in item.get("evidence", []):
        evidence.append({
            "message_id": m.get("message_id"),
            "date": m.get("date"),
            "channel": m.get("channel"),
            "text_snippet": m.get("text", "")[:200]
        })

    return {
        "account_id": item["account_id"],
        "risk_level": item["risk_label"],
        "explanation": explanation,
        "evidence_json": json.dumps(evidence, ensure_ascii=False),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }


def upsert_flag(conn, flag: dict):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO risk_flags (id, account_id, risk_level, explanation, evidence_json, created_at)"
        " VALUES ((SELECT id FROM risk_flags WHERE account_id = ?), ?, ?, ?, ?, ?)",
        (
            flag["account_id"],
            flag["account_id"],
            flag["risk_level"],
            flag["explanation"],
            flag["evidence_json"],
            flag["created_at"],
        ),
    )
    conn.commit()


def list_flags(conn) -> List[dict]:
    cur = conn.cursor()
    cur.execute("SELECT account_id, risk_level, explanation, created_at FROM risk_flags ORDER BY created_at DESC")
    rows = cur.fetchall()
    return [
        {"account_id": r[0], "risk_level": r[1], "explanation": r[2], "created_at": r[3]} for r in rows
    ]


def get_flag(conn, account_id: str) -> dict:
    """Return the full flag record for a single account, including evidence parsed from JSON."""
    cur = conn.cursor()
    cur.execute("SELECT account_id, risk_level, explanation, evidence_json, created_at FROM risk_flags WHERE account_id = ?", (account_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "account_id": row[0],
        "risk_level": row[1],
        "explanation": row[2],
        "evidence": json.loads(row[3]) if row[3] else [],
        "created_at": row[4]
    }


def generate_and_persist_flags(messages, bm25, collection, conn):
    ensure_flags_table(conn)

    scanned = agent.scan_accounts(messages, bm25, collection, conn)
    if not scanned:
        print("No flagged accounts from agent scanner.")
        return []

    persisted = []
    for item in scanned:
        flag = make_flag_record(item)
        upsert_flag(conn, flag)
        persisted.append(flag)

    return persisted


def main():
    # Load messages and construct BM25 index (re-uses Phase 1-6 artifacts)
    messages = agent.load_messages()
    texts = [m["text"] for m in messages]
    bm25 = BM25 = None
    try:
        from hybrid_search_eval import BM25 as BM25Cls
        BM25 = BM25Cls(texts)
    except Exception:
        print("Warning: BM25 not available; hybrid searches will be skipped.")
        BM25 = None

    # Try to open chroma if available (agent.scan_accounts will handle fallback)
    collection = None
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = agent.safe_get_collection(chroma_client, "client_messages", sentence_transformer_ef)
    except Exception:
        collection = None

    conn = init_db()

    persisted = generate_and_persist_flags(messages, BM25, collection, conn)
    if persisted:
        print(f"Persisted {len(persisted)} flags:")
        for p in persisted:
            print(f" - {p['account_id']}: {p['risk_level']} ({p['created_at']})")
    else:
        print("No flags persisted.")

    print("\nCurrent flags in DB:")
    for r in list_flags(conn):
        print(f" - {r['account_id']}: {r['risk_level']} — {r['explanation']} @ {r['created_at']}")

    conn.close()


if __name__ == "__main__":
    main()
