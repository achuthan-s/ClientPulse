"""
Phase 8: Agent (simple, local implementation)
---------------------------------------------
Provides a lightweight agent that exposes the tools the later LangGraph
agent would use. This stays offline-friendly (no external LLMs) and uses
existing modules (`trend_tracker`, `hybrid_search_eval`, `rag_copilot`) to
decide which accounts merit further review.

Tools provided:
 - get_account_history(account_id): list of messages for that account
 - get_recent_messages(account_id, n): last `n` messages
 - calculate_sentiment_trend(account_id): calls `compute_account_trend`
 - search_messages(query, account_id, top_k): hybrid search with BM25 fallback

Agent behavior (simple rule-based): For each account, compute trend metrics
and count recent scope dispute signals. Flag accounts if:
 - `classify_risk` returns WATCH or HIGH RISK; OR
 - >= 2 scope dispute messages in the last 5 messages; OR
 - slope < -0.02 AND drift < -0.15

The agent returns a list of candidate accounts with the triggering signals
and a short set of evidence messages (dates + snippets).
"""

import json
import os
from typing import List, Dict

from hybrid_search_eval import BM25, hybrid_search
from trend_tracker import init_db, populate_db, get_all_account_ids, compute_account_trend, classify_risk

# Reuse safe collection helper from rag_copilot to be consistent
from rag_copilot import safe_get_collection


def load_messages(path="data/extracted_messages.json") -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run Phase 1-2 first.")
    with open(path, "r") as f:
        return json.load(f)


def get_account_history(messages: List[dict], account_id: str) -> List[dict]:
    return sorted([m for m in messages if m["account_id"] == account_id], key=lambda m: m["date"])


def get_recent_messages(messages: List[dict], account_id: str, n: int = 5) -> List[dict]:
    hist = get_account_history(messages, account_id)
    return hist[-n:]


def search_messages(query: str, messages: List[dict], bm25: BM25, collection, account_id: str = None, top_k: int = 5):
    # If collection is None, fall back to BM25-only search
    if collection is None:
        try:
            bm25_results = bm25.search(query, top_k=top_k)
        except Exception:
            return []
        if account_id:
            bm25_results = [(i, s) for (i, s) in bm25_results if messages[i]["account_id"] == account_id]
        return [messages[i].copy() for i, _ in bm25_results]
    else:
        return hybrid_search(query, messages, bm25, collection, top_k=top_k, account_id=account_id)


def scan_accounts(messages: List[dict], bm25: BM25, collection, conn) -> List[Dict]:
    results = []
    account_ids = get_all_account_ids(conn)

    for acct in sorted(account_ids):
        trend = compute_account_trend(conn, acct)
        if not trend:
            continue

        risk_label = classify_risk(trend)

        recent = get_recent_messages(messages, acct, n=5)
        recent_scope_count = sum(1 for m in recent if m.get("scope_dispute_signal"))

        # Simple rule-based decision
        flagged = False
        reasons = []

        if "HIGH RISK" in risk_label:
            flagged = True
            reasons.append("High risk by trend classifier")
        elif "WATCH" in risk_label:
            flagged = True
            reasons.append("Watch by trend classifier")

        if recent_scope_count >= 2:
            flagged = True
            reasons.append(f"{recent_scope_count} scope dispute messages in last 5")

        if trend.get("slope", 0) < -0.02 and trend.get("drift", 0) < -0.15:
            flagged = True
            reasons.append("Sustained negative slope and significant drift")

        if flagged:
            # Gather evidence: top negative messages and scope dispute messages
            evidence = sorted([m for m in get_account_history(messages, acct) if m.get("sentiment") <= 0], key=lambda m: m["date"])[:5]
            # If not enough, run a focused search for scope/delay/payment keywords
            if len(evidence) < 3:
                extra = search_messages("scope OR delay OR payment OR cancel OR outage", messages, bm25, collection, account_id=acct, top_k=3)
                for e in extra:
                    if e not in evidence:
                        evidence.append(e)

            results.append({
                "account_id": acct,
                "risk_label": risk_label,
                "reasons": reasons,
                "trend": trend,
                "evidence": evidence
            })

    return results


def main():
    messages = load_messages()
    texts = [m["text"] for m in messages]
    bm25 = BM25(texts)

    # Initialize collection if possible
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = safe_get_collection(chroma_client, "client_messages", sentence_transformer_ef)
    except Exception:
        collection = None

    conn = init_db()
    populate_db(conn, messages)

    print("Scanning accounts with agent rules...\n")
    flagged = scan_accounts(messages, bm25, collection, conn)

    if not flagged:
        print("No accounts flagged by the agent rules.")
    else:
        for item in flagged:
            print(f"Account: {item['account_id']} — {item['risk_label']}")
            print(f"  Reasons: {', '.join(item['reasons'])}")
            print("  Evidence:")
            for e in item["evidence"]:
                print(f"    {e['date'][:10]} | {e['channel']:12s} | {e['text'][:80]}")
            print("")

    conn.close()


if __name__ == "__main__":
    main()
