"""
Phase 6: Hybrid Search & Evaluation
-------------------------------------
Part A — Hybrid Search:
  Combines BM25 keyword search with dense vector search, then merges
  results using Reciprocal Rank Fusion (RRF). This gives us the best
  of both worlds: exact term matching AND semantic similarity.

Part B — Evaluation:
  Uses our labeled synthetic accounts (we know ground truth) to measure
  whether the trend system correctly flags the right accounts and roughly
  when it would have detected the decline.

WHY HYBRID SEARCH?
------------------
Embedding models understand meaning but can miss exact tokens:
  - Query "SLA breach penalty" might not surface a message that literally
    says "SLA breach" if the embedding is distracted by other words.
BM25 (Best Match 25) treats search like a keyword matching problem:
  - It directly rewards documents containing the EXACT query terms,
    weighted by how rare those terms are across the corpus (TF-IDF logic).
Combining both means: semantically similar results PLUS exact term hits.
"""

import json
import math
import sqlite3
import os
from collections import defaultdict
from typing import List, Dict, Tuple

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

# ---------------------------------------------------------------------------
# PART A: BM25 IMPLEMENTATION
# ---------------------------------------------------------------------------

class BM25:
    """
    A from-scratch BM25 implementation so you can see exactly how it works.
    
    BM25 scores how relevant a document is to a query based on:
      - Term Frequency (TF): how often the query term appears in this document
      - Inverse Document Frequency (IDF): how rare the term is across all docs
        (rare terms that appear in a specific doc are strong signals)
      - Document length normalization: prevents long documents from always winning

    Parameters:
      k1 (float): Controls TF saturation. Higher k1 = TF keeps mattering more
                  at high frequencies. Typically 1.2–2.0.
      b  (float): Controls length normalization. b=1.0 = full normalization,
                  b=0.0 = no normalization. Typically 0.75.
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # Tokenize: lowercase, split on whitespace (simple but effective for English)
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.n_docs = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized_corpus) / self.n_docs
        self.df = self._compute_df()       # document frequency per term
        self.idf = self._compute_idf()    # IDF score per term

    def _compute_df(self) -> Dict[str, int]:
        """Count how many documents contain each term."""
        df = defaultdict(int)
        for doc in self.tokenized_corpus:
            for term in set(doc):  # set() so we count each term once per doc
                df[term] += 1
        return df

    def _compute_idf(self) -> Dict[str, float]:
        """
        IDF formula (Robertson-Sparck Jones variant used in BM25):
          log( (N - df + 0.5) / (df + 0.5) + 1 )
        
        A term that appears in every doc has IDF ≈ 0 (uninformative).
        A term that appears in only 1 doc has IDF ≈ log(N) (very informative).
        """
        idf = {}
        for term, df in self.df.items():
            idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
        return idf

    def score(self, query: str, doc_index: int) -> float:
        """Score a single document against a query."""
        query_terms = query.lower().split()
        doc = self.tokenized_corpus[doc_index]
        doc_len = len(doc)

        # Term frequency count for this document
        tf_counts = defaultdict(int)
        for term in doc:
            tf_counts[term] += 1

        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue  # Term not seen in corpus → skip
            tf = tf_counts[term]
            # BM25 term score formula
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf[term] * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top_k (index, score) pairs, sorted by score descending."""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        # Filter out zero-score docs (no query terms found at all)
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# RECIPROCAL RANK FUSION
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    RRF merges multiple ranked result lists into one combined ranking.

    For each document, its RRF score = Σ  1 / (k + rank_in_list_i)
    across all lists. k=60 is the standard constant from the original paper
    (prevents very top-ranked docs from dominating too strongly).

    WHY RRF vs. simple score averaging?
    - BM25 and embedding scores live on completely different scales.
      BM25 might return [12.4, 8.1, 3.2], embeddings [0.87, 0.81, 0.76].
      Averaging those raw numbers would be meaningless.
    - RRF only uses the RANK (position), not the raw score, so the different
      scales don't matter at all.
    """
    rrf_scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_idx] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# HYBRID SEARCH FUNCTION
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    messages: List[dict],
    bm25: BM25,
    chroma_collection,
    top_k: int = 5,
    account_id: str = None
) -> List[dict]:
    """
    Runs BM25 + vector search in parallel, merges via RRF, returns top_k results.
    Optionally filters by account_id before searching.
    """
    # --- BM25 leg ---
    bm25_results = bm25.search(query, top_k=top_k * 2)  # fetch extra for merging

    # If filtering by account, mask out messages from other accounts
    if account_id:
        bm25_results = [
            (i, s) for (i, s) in bm25_results
            if messages[i]["account_id"] == account_id
        ]

    # --- Vector search leg ---
    chroma_kwargs = {
        "query_texts": [query],
        "n_results": min(top_k * 2, len(messages))
    }
    if account_id:
        chroma_kwargs["where"] = {"account_id": account_id}

    vector_results_raw = chroma_collection.query(**chroma_kwargs)

    # Map chroma results back to our message index positions
    # ChromaDB returns IDs; we need to look them up in our message list
    msg_id_to_idx = {msg["message_id"]: i for i, msg in enumerate(messages)}
    vector_results = [
        (msg_id_to_idx[mid], 1.0 - dist)  # convert distance → similarity score
        for mid, dist in zip(
            vector_results_raw["ids"][0],
            vector_results_raw["distances"][0]
        )
        if mid in msg_id_to_idx
    ]

    # --- Fuse ---
    fused = reciprocal_rank_fusion([bm25_results, vector_results])

    # Return the top_k fused results as full message dicts
    top_results = []
    for doc_idx, rrf_score in fused[:top_k]:
        result = messages[doc_idx].copy()
        result["_rrf_score"] = round(rrf_score, 4)
        top_results.append(result)
    return top_results


# ---------------------------------------------------------------------------
# PART B: EVALUATION HARNESS
# ---------------------------------------------------------------------------

# Ground truth: what we designed into the synthetic data
GROUND_TRUTH = {
    "should_be_flagged":     ["acct_techcorp", "acct_dataflow"],
    "should_not_be_flagged": ["acct_innovate", "acct_cloudsync"],
    # At which MESSAGE INDEX did TechCorp's decline meaningfully start?
    # (msg index 4 = the first scope dispute email, 0-indexed)
    "techcorp_decline_starts_at_index": 4,
}

def run_evaluation(messages: List[dict]):
    """
    Evaluates two things:
    1. CLASSIFICATION: Does the trend system correctly flag the right accounts?
    2. TIMING: For TechCorp, does it detect the decline reasonably early?
    """
    from trend_tracker import init_db, populate_db, get_all_account_ids
    from trend_tracker import compute_account_trend, classify_risk

    print("\n" + "=" * 60)
    print("  EVALUATION HARNESS")
    print("=" * 60)

    conn = init_db()
    populate_db(conn, messages)

    # --- 1. Classification Accuracy ---
    correct = 0
    total = len(GROUND_TRUTH["should_be_flagged"]) + len(GROUND_TRUTH["should_not_be_flagged"])

    print("\n[Classification Accuracy]")
    for acct in get_all_account_ids(conn):
        trend  = compute_account_trend(conn, acct)
        risk   = classify_risk(trend)
        flagged = "HIGH RISK" in risk or "WATCH" in risk

        if acct in GROUND_TRUTH["should_be_flagged"]:
            expected_flag = True
        else:
            expected_flag = False

        match = flagged == expected_flag
        correct += 1 if match else 0
        status = "✅ CORRECT" if match else "❌ WRONG"
        print(f"  {acct:25s} flagged={flagged}, expected={expected_flag}  {status}")

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}")

    # --- 2. Timing: When does TechCorp trip the wire? ---
    print("\n[Early Detection — TechCorp]")
    techcorp_msgs = sorted(
        [m for m in messages if m["account_id"] == "acct_techcorp"],
        key=lambda m: m["date"]
    )

    # Simulate the system seeing messages one at a time (streaming scenario)
    for i in range(2, len(techcorp_msgs) + 1):
        subset = techcorp_msgs[:i]
        conn2 = sqlite3.connect(":memory:")  # fresh in-memory DB for each window
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
        trend = compute_account_trend(conn2, "acct_techcorp")
        risk  = classify_risk(trend)
        flagged = "HIGH RISK" in risk or "WATCH" in risk
        conn2.close()

        if flagged:
            designed_trigger = GROUND_TRUTH["techcorp_decline_starts_at_index"]
            msgs_after_trigger = i - 1 - designed_trigger
            print(f"  ⚡ System flagged TechCorp after message #{i-1} ({subset[-1]['date'][:10]})")
            print(f"     That is {msgs_after_trigger} message(s) after the designed trigger point.")
            break
    else:
        print("  ⚠️  System NEVER flagged TechCorp — something is wrong.")

    conn.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    with open("data/extracted_messages.json") as f:
        messages = json.load(f)

    texts = [m["text"] for m in messages]

    print("Building BM25 index...")
    bm25 = BM25(texts)

    # Connect to the ChromaDB we built in Phase 4
    HF_HOME_PATH = os.path.join(os.path.dirname(__file__), ".cache", "huggingface")
    os.environ.setdefault("HF_HOME", HF_HOME_PATH)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(
        name="client_messages",
        embedding_function=sentence_transformer_ef
    )

    # --- Hybrid Search Demo ---
    query = "SLA breach penalty clause"
    print(f"\n{'='*60}")
    print(f"Hybrid Search — Query: '{query}'")
    print(f"{'='*60}")

    # BM25-only (to show what pure keyword search finds)
    print("\n[BM25 only]")
    bm25_top = bm25.search(query, top_k=3)
    for idx, score in bm25_top:
        m = messages[idx]
        print(f"  Score {score:.2f} | [{m['account_id']}] {m['text']}")

    # Hybrid
    print("\n[Hybrid BM25 + Vector (RRF merged)]")
    hybrid_top = hybrid_search(query, messages, bm25, collection, top_k=3)
    for r in hybrid_top:
        print(f"  RRF {r['_rrf_score']} | [{r['account_id']}] {r['text']}")

    # --- Evaluation ---
    run_evaluation(messages)


if __name__ == "__main__":
    main()
