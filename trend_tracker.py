"""
Phase 5: Trend Tracking
-----------------------
This module builds a SQLite persistence layer to store per-account sentiment
over time, then computes trends (rolling average + linear slope) to detect
which accounts are declining, stable, or recovering.

KEY DESIGN INSIGHT:
We compare each account against ITS OWN baseline, not a global threshold.
A sentiment score of 0.1 means something very different for an account that
normally sits at 0.8 (alarming drop) vs. one that normally sits at 0.0 (normal).
Comparing to a fixed global threshold would cause constant false positives/negatives.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. DATABASE LAYER
# ---------------------------------------------------------------------------

DB_PATH = "data/clientpulse.db"

def init_db():
    """
    Create the SQLite database and the sentiment_history table if they
    don't already exist. SQLite is perfect here: it's zero-configuration,
    requires no server, and the data is just a single file.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Each row = one message's extracted sentiment tied to an account and a date.
    # This is our time-series table: append-only, ordered by date.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id          TEXT UNIQUE,       -- prevent duplicate inserts
            account_id          TEXT NOT NULL,
            date                TEXT NOT NULL,      -- ISO-8601 string for easy ordering
            sentiment           REAL NOT NULL,      -- float from -1.0 to 1.0
            scope_dispute_signal INTEGER NOT NULL,  -- SQLite has no BOOLEAN; 0 or 1
            channel             TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def populate_db(conn: sqlite3.Connection, messages: List[dict]):
    """
    Insert messages into the database. We use INSERT OR IGNORE so this is
    idempotent: safe to run multiple times without creating duplicates.
    """
    cursor = conn.cursor()
    for msg in messages:
        cursor.execute("""
            INSERT OR IGNORE INTO sentiment_history
                (message_id, account_id, date, sentiment, scope_dispute_signal, channel)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            msg["message_id"],
            msg["account_id"],
            msg["date"],
            float(msg["sentiment"]),
            1 if msg.get("scope_dispute_signal") else 0,
            msg["channel"]
        ))
    conn.commit()
    print(f"Inserted/verified {len(messages)} records in the database.")


def get_sentiment_series(conn: sqlite3.Connection, account_id: str) -> List[Tuple]:
    """
    Retrieve the full chronological sentiment history for a single account.
    Returns a list of (date, sentiment) tuples, ordered by date ascending.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, sentiment
        FROM sentiment_history
        WHERE account_id = ?
        ORDER BY date ASC
    """, (account_id,))
    return cursor.fetchall()


def get_all_account_ids(conn: sqlite3.Connection) -> List[str]:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT account_id FROM sentiment_history")
    return [row[0] for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# 2. TREND COMPUTATION
# ---------------------------------------------------------------------------

def rolling_average(sentiments: List[float], window: int = 3) -> List[float]:
    """
    Computes a simple moving average over a list of sentiment scores.

    WHY: A single angry email (-0.9) can be noise: maybe the client was having
    a bad day. A rolling average smooths these outliers so we only see
    sustained trends, not momentary spikes.

    Example with window=3:
        Input:  [0.5, 0.4, 0.3, 0.2, -0.1]
        Output: [0.5, 0.45, 0.4, 0.3, 0.13]  (first windows use fewer points)
    """
    result = []
    for i in range(len(sentiments)):
        # Use a partial window at the start (can't average before we have data)
        window_slice = sentiments[max(0, i - window + 1): i + 1]
        result.append(sum(window_slice) / len(window_slice))
    return result


def compute_slope(sentiments: List[float]) -> float:
    """
    Computes the linear slope of sentiment over time using the least-squares
    formula. This is the single best number to answer: "Is this account
    trending up or down, and how fast?"

    A slope of -0.05 means: on average, each new message is 0.05 points
    more negative than the previous one — a clear decline signal.

    WHY NOT JUST COMPARE FIRST VS LAST MESSAGE?
    Comparing only the first and last points is extremely noisy. One unusually
    happy final message could hide a month of decline before it. A slope over
    all N points is much more robust.
    """
    n = len(sentiments)
    if n < 2:
        return 0.0  # Can't compute a slope with fewer than 2 points

    # x-axis = message index (0, 1, 2, ..., n-1)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(sentiments) / n

    # Least-squares slope formula: Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    numerator   = sum((x[i] - x_mean) * (sentiments[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    return numerator / denominator if denominator != 0 else 0.0


def compute_account_trend(conn: sqlite3.Connection, account_id: str, window: int = 4):
    """
    Pulls the full sentiment history for an account and computes:
      - baseline: the account's own average sentiment (NOT a global threshold)
      - recent_avg: the rolling average of the last `window` messages
      - slope: the linear trend direction over all messages
      - drift: how far the recent average has drifted from the baseline
    """
    series = get_sentiment_series(conn, account_id)
    if not series:
        return None

    dates      = [row[0] for row in series]
    sentiments = [row[1] for row in series]
    smoothed   = rolling_average(sentiments, window=window)
    slope      = compute_slope(sentiments)

    # Baseline = account's own historical average (first half of history)
    # WHY: We use only the first half as baseline so the baseline isn't
    # polluted by a recent decline we're trying to detect.
    half = max(1, len(sentiments) // 2)
    baseline    = sum(sentiments[:half]) / half
    recent_avg  = smoothed[-1]  # The smoothed value of the most recent message
    drift       = recent_avg - baseline   # Negative = declining from its own baseline

    return {
        "account_id":  account_id,
        "n_messages":  len(sentiments),
        "baseline":    round(baseline, 3),
        "recent_avg":  round(recent_avg, 3),
        "slope":       round(slope, 4),
        "drift":       round(drift, 3),
        "first_date":  dates[0][:10],
        "last_date":   dates[-1][:10],
        "smoothed_series": [round(s, 3) for s in smoothed],
    }


def classify_risk(trend: dict) -> str:
    """
    Simple rule-based risk classifier using the computed trend metrics.
    In Phase 8, the agent will replace this with autonomous reasoning.

    WHY MULTIPLE SIGNALS?
    A negative slope alone could just be one rough week. We want to see:
    - A sustained slope (direction over time)
    - AND a drift from the account's own baseline (not just from zero)
    Requiring both reduces false positives significantly.
    """
    slope = trend["slope"]
    drift = trend["drift"]

    if slope < -0.03 and drift < -0.2:
        return "🔴 HIGH RISK"
    elif slope < -0.01 or drift < -0.1:
        return "🟡 WATCH"
    elif slope > 0.01 and drift > 0.05:
        return "🟢 IMPROVING"
    else:
        return "🟢 STABLE"


# ---------------------------------------------------------------------------
# 3. MAIN REPORT
# ---------------------------------------------------------------------------

def main():
    print("Loading extracted messages...")
    with open("data/extracted_messages.json") as f:
        messages = json.load(f)

    print("Initializing database...")
    conn = init_db()
    populate_db(conn, messages)

    print("\n" + "=" * 60)
    print("  CLIENTPULSE — ACCOUNT TREND REPORT")
    print("=" * 60)

    for account_id in sorted(get_all_account_ids(conn)):
        trend = compute_account_trend(conn, account_id)
        if not trend:
            continue

        risk   = classify_risk(trend)
        print(f"\nAccount : {account_id}")
        print(f"Risk    : {risk}")
        print(f"Period  : {trend['first_date']} → {trend['last_date']} ({trend['n_messages']} messages)")
        print(f"Baseline (own avg, first half) : {trend['baseline']:+.3f}")
        print(f"Recent  (rolling avg)          : {trend['recent_avg']:+.3f}")
        print(f"Drift from baseline            : {trend['drift']:+.3f}")
        print(f"Slope (trend direction)        : {trend['slope']:+.4f}")
        print(f"Smoothed series: {trend['smoothed_series']}")

    print("\n" + "=" * 60)
    conn.close()


if __name__ == "__main__":
    main()
