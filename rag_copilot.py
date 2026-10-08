"""
Phase 7: RAG Copilot
---------------------
Answers the question "Why is Account X flagged?" by:
  1. Retrieving the most relevant messages for that account using hybrid search
  2. Assembling those messages into a prompt as evidence (the "context")
  3. Calling an LLM that is instructed to answer ONLY from that context,
     cite specific messages by date, and say "I don't have enough evidence"
     if the context doesn't support a claim.

This is the core RAG (Retrieval-Augmented Generation) pattern:
  Retrieve → Augment the prompt → Generate a grounded answer.

WHY RAG INSTEAD OF JUST ASKING THE LLM?
----------------------------------------
If you ask an LLM "Why is TechCorp unhappy?", it will confabulate a
plausible-sounding answer from training data — about a company it has
never seen. RAG forces the model to work only from the actual retrieved
evidence, making answers verifiable and traceable.

THE "REFUSE TO GUESS" RULE:
----------------------------
This is critical for a trust system. If a user asks about something not
in the context (e.g., "Did the CEO mention budget cuts?"), a RAG system
without guardrails will still hallucinate an answer. Our system prompt
explicitly instructs the model to say "I don't have enough evidence"
rather than speculate. This is enforced by prompt engineering, not code.
"""

import json
import os
import sys
from typing import List

import requests

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

# Import our previously built modules
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_search_eval import BM25, hybrid_search
from trend_tracker import init_db, populate_db, compute_account_trend, classify_risk

# ---------------------------------------------------------------------------
# 1. THE SYSTEM PROMPT — The most important engineering decision in RAG
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are ClientPulse Copilot, an AI assistant that helps
account managers understand why a client account is flagged for risk.

CRITICAL RULES — follow them exactly:
1. Answer ONLY using the evidence provided in the [CONTEXT] section below.
2. Every claim you make MUST be cited using the format: [DATE | CHANNEL].
   Example: "The client mentioned contract termination [2023-04-14 | email]."
3. If the context does not contain enough information to answer the question,
   respond with: "I don't have enough evidence in the retrieved messages to
   answer this confidently. Consider broadening the search."
4. Do NOT use any prior knowledge. Do NOT guess or infer beyond the evidence.
5. Keep your answer concise: 3–5 sentences maximum, with inline citations.
"""
RAG_SYSTEM_PROMPT += "\nWhen citing, use exactly the bracketed [DATE | CHANNEL] tokens that\nappear at the start of each evidence item in the [CONTEXT] section. Do not\nuse numeric indices or other shorthand for citations."

# ---------------------------------------------------------------------------
# 2. RETRIEVAL — Pull the most relevant messages for an account
# ---------------------------------------------------------------------------

def retrieve_evidence(
    query: str,
    account_id: str,
    messages: List[dict],
    bm25: BM25,
    collection,
    top_k: int = 6
) -> List[dict]:
    """
    Uses the hybrid search from Phase 6 to retrieve the most relevant
    messages for a specific account given the user's question.

    WHY top_k=6?
    LLM context windows are not infinite, and more context ≠ better answers.
    Research (the "lost in the middle" paper) shows LLMs often ignore
    evidence buried in the middle of very long contexts. 5–8 high-quality
    retrieved chunks is typically better than 20 mediocre ones.
    """
    # If Chroma collection not available, fall back to BM25-only search.
    if collection is None:
        # BM25 returns (index, score) pairs — map them back to messages
        try:
            bm25_results = bm25.search(query, top_k=top_k)
        except Exception:
            return []

        if account_id:
            bm25_results = [(i, s) for (i, s) in bm25_results if messages[i]["account_id"] == account_id]

        return [messages[i].copy() for i, _ in bm25_results]

    return hybrid_search(
        query=query,
        messages=messages,
        bm25=bm25,
        chroma_collection=collection,
        top_k=top_k,
        account_id=account_id
    )


def safe_get_collection(client, name: str, embedding_function):
    """Try to get an existing Chroma collection; return None with a warning on failure."""
    try:
        return client.get_collection(name=name, embedding_function=embedding_function)
    except Exception as e:
        print(f"[Warning] Could not open Chroma collection '{name}': {e}. Falling back to BM25-only retrieval.")
        return None


# ---------------------------------------------------------------------------
# 3. PROMPT ASSEMBLY — Build the augmented prompt from retrieved evidence
# ---------------------------------------------------------------------------

def build_rag_prompt(query: str, account_id: str, evidence: List[dict], trend: dict) -> str:
    """
    Assembles the full prompt by injecting:
      - The account trend summary (for high-level context)
      - The retrieved messages (the actual evidence)
      - The user's question

    WHY SEPARATE SECTIONS?
    Clearly delimited sections ([CONTEXT], [QUESTION]) help the LLM
    understand the structure of the input. Research shows models are
    more accurate when the context is clearly separated from the question.
    """
    # Format each retrieved message as a citable evidence item. We place an
    # explicit bracketed token [DATE | CHANNEL] at the start of each item so
    # the LLM can cite exactly the required form (see system prompt).
    context_lines = []
    for i, msg in enumerate(evidence, 1):
        date = msg["date"][:10]
        channel = msg["channel"]
        text = msg["text"]
        sentiment = msg.get("sentiment", "N/A")
        scope = "⚠️ scope dispute" if msg.get("scope_dispute_signal") else ""
        # The bracketed token below is the canonical citation token the model
        # must use when referencing this message in its answer.
        context_lines.append(
            f"[{date} | {channel}] (evidence #{i}) Sentiment: {sentiment} {scope}\n"
            f"    \"{text}\""
        )

    context_block = "\n\n".join(context_lines)

    risk_summary = (
        f"Account: {account_id}\n"
        f"Risk classification: {trend.get('risk', 'Unknown')}\n"
        f"Trend slope: {trend.get('slope', 0):+.4f} | "
        f"Drift from baseline: {trend.get('drift', 0):+.3f}\n"
        f"Period: {trend.get('first_date', '?')} → {trend.get('last_date', '?')} "
        f"({trend.get('n_messages', '?')} messages)"
    )

    return f"""[ACCOUNT SUMMARY]
{risk_summary}

[CONTEXT — Retrieved evidence for your question]
{context_block}

[QUESTION]
{query}"""


# ---------------------------------------------------------------------------
# 4. LLM CALL — Generate a grounded answer
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls a configured LLM provider with our system prompt + assembled RAG prompt.
    Prefers Grok, then NVIDIA, then Gemini, and otherwise returns a safe mock
    response so local demos remain usable.
    """
    grok_api_key = os.environ.get("GROK_API_KEY")
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    # Prefer Grok because it is a common alternative to Gemini in the current app flow.
    if grok_api_key:
        try:
            model = os.environ.get("GROK_MODEL", "grok-2-latest")
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {grok_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Grok API call failed: {e}]"

    if nvidia_api_key:
        try:
            model = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {nvidia_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[NVIDIA API call failed: {e}]"

    if gemini_api_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                    "temperature": 0.2,
                }
            )
            return response.text
        except Exception as e:
            return f"[Gemini API call failed: {e}]"

    return (
        "[MOCK — No API key configured. Here is what the LLM would receive:]\n\n"
        f"SYSTEM:\n{system_prompt}\n\n"
        f"USER:\n{user_prompt}\n\n"
        "[The LLM would respond with a grounded, cited answer based on the context above.]"
    )


# ---------------------------------------------------------------------------
# 5. THE COPILOT FUNCTION — The public API for Phases 9 and 10
# ---------------------------------------------------------------------------

def ask_copilot(
    question: str,
    account_id: str,
    messages: List[dict],
    bm25: BM25,
    collection,
    conn
) -> dict:
    """
    The main entry point. Given a question about an account, returns:
      - answer: the grounded LLM response with citations
      - evidence: the retrieved source messages
      - trend: the account's computed trend data

    This function will be called directly from the FastAPI backend in Phase 10.
    """
    # Get the account's trend data to include in the prompt
    trend_data = compute_account_trend(conn, account_id) or {}
    trend_data["risk"] = classify_risk(trend_data) if trend_data else "Unknown"

    # Retrieve the most relevant evidence chunks
    evidence = retrieve_evidence(
        query=question,
        account_id=account_id,
        messages=messages,
        bm25=bm25,
        collection=collection
    )

    if not evidence:
        return {
            "answer": "I don't have enough evidence in the retrieved messages to answer this confidently.",
            "evidence": [],
            "trend": trend_data
        }

    # Build the augmented prompt
    user_prompt = build_rag_prompt(question, account_id, evidence, trend_data)

    # Generate the grounded answer
    answer = call_llm(RAG_SYSTEM_PROMPT, user_prompt)

    return {
        "answer": answer,
        "evidence": evidence,
        "trend": trend_data,
        "prompt_sent": user_prompt  # Useful for debugging / transparency
    }


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------

def main():
    # Load data and initialize all components
    with open("data/extracted_messages.json") as f:
        messages = json.load(f)

    texts = [m["text"] for m in messages]
    bm25 = BM25(texts)

    # Embedding function and Chroma are optional for offline demos. If the
    # packages aren't installed, we proceed with BM25-only retrieval by
    # leaving `collection` as None.
    if embedding_functions is not None and chromadb is not None:
        os.environ.setdefault(
            "HF_HOME",
            os.path.join(os.path.dirname(__file__), ".cache", "huggingface")
        )
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = safe_get_collection(chroma_client, "client_messages", sentence_transformer_ef)
    else:
        sentence_transformer_ef = None
        collection = None
    conn = init_db()
    populate_db(conn, messages)

    # --- Demo Question 1: Why is TechCorp flagged? ---
    print("\n" + "=" * 65)
    print("COPILOT DEMO — Q1")
    print("=" * 65)
    q1 = "Why is this account flagged and what are the specific concerns?"
    print(f"Account : acct_techcorp")
    print(f"Question: {q1}\n")

    result1 = ask_copilot(q1, "acct_techcorp", messages, bm25, collection, conn)

    print("[Evidence Retrieved]")
    for e in result1["evidence"]:
        print(f"  {e['date'][:10]} | {e['channel']:15s} | {e['text'][:70]}...")

    print("\n[Copilot Answer]")
    print(result1["answer"])

    # --- Demo Question 2: Question about something NOT in the evidence ---
    print("\n" + "=" * 65)
    print("COPILOT DEMO — Q2 (Testing the 'refuse to guess' rule)")
    print("=" * 65)
    q2 = "Did the client's CFO personally request a refund?"
    print(f"Account : acct_techcorp")
    print(f"Question: {q2}\n")

    result2 = ask_copilot(q2, "acct_techcorp", messages, bm25, collection, conn)
    print("[Copilot Answer]")
    print(result2["answer"])

    # Run offline refusal test (no GEMINI API key required)
    run_offline_refusal_test(messages, bm25, collection, conn)

def run_offline_refusal_test(messages, bm25, collection, conn):
    """Simple offline test: ask about a non-existent account and verify refusal."""
    print("\n[Offline Refusal Test]")
    q = "Did the client's CFO personally request a refund?"
    # Use a made-up account to ensure no evidence is returned
    result = ask_copilot(q, "acct_nonexistent", messages, bm25, collection, conn)
    print("Question:", q)
    print("Answer:", result["answer"])
    if "I don't have enough evidence" in result["answer"]:
        print("Offline refusal behavior: OK")
    else:
        print("Offline refusal behavior: NOT OK — check retrieval or mocking logic")

    # Show the assembled prompt that was sent for transparency/debugging
    print("\n" + "=" * 65)
    print("TRANSPARENCY — Prompt sent to LLM for this offline query")
    print("=" * 65)
    print(result.get("prompt_sent", ""))

    # Do NOT close the DB connection here; leave lifecycle to the caller.


if __name__ == "__main__":
    main()
