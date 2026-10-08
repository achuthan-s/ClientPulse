import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  Tooltip,
  ReferenceLine,
} from "recharts";
import api from "../api";

// ---------------------------------------------------------------------------
// SPARKLINE — tiny inline chart, one per account row.
// Shows the raw sentiment dots + smoothed trend line so a viewer can see
// the shape of the decline at a glance without opening the detail page.
// We intentionally suppress axes and gridlines to keep it minimal.
// ---------------------------------------------------------------------------
function Sparkline({ series, baseline, riskLevel }) {
  if (!series || series.length === 0) return null;

  // Pick line color to match the risk level so the chart reinforces the badge
  const lineColor =
    riskLevel.includes("HIGH") ? "var(--danger)" :
    riskLevel.includes("WATCH") ? "var(--warning)" :
    riskLevel.includes("IMPROVING") ? "var(--success)" : "var(--primary)";

  return (
    <div className="sparkline-wrap" aria-label="Sentiment trend sparkline">
      <ResponsiveContainer width="100%" height={48}>
        <LineChart data={series} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          {/* Baseline reference line — the account's own historical average.
              This is the key signal: crossing below YOUR own line, not a
              global threshold. */}
          <ReferenceLine
            y={baseline}
            stroke="rgba(148,163,184,0.35)"
            strokeDasharray="3 3"
          />
          {/* Raw sentiment as faint dots so the volatility is visible */}
          <Line
            type="monotone"
            dataKey="raw"
            stroke={lineColor}
            strokeWidth={0}
            dot={{ r: 2, fill: lineColor, opacity: 0.4 }}
            isAnimationActive={false}
          />
          {/* Smoothed (rolling average) as the primary line — this is the
              trend signal, not the noise of individual messages */}
          <Line
            type="monotone"
            dataKey="smoothed"
            stroke={lineColor}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {/* Minimal tooltip on hover: just show the date and smoothed value */}
          <Tooltip
            contentStyle={{
              background: "var(--bg-soft)",
              border: "1px solid var(--line)",
              borderRadius: 8,
              fontSize: 11,
              color: "var(--text)",
            }}
            formatter={(v, name) => [v?.toFixed(2), name === "smoothed" ? "Trend" : "Raw"]}
            labelFormatter={(l) => series[l]?.date ?? ""}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RISK BADGE — color-coded pill that mirrors the Python classify_risk output.
// The emoji prefix (🔴/🟡/🟢) in risk_level drives the CSS class so the
// badge color is always consistent with the backend classification.
// ---------------------------------------------------------------------------
function RiskBadge({ level }) {
  const cls =
    level.includes("HIGH") ? "risk-badge risk-high" :
    level.includes("WATCH") ? "risk-badge risk-watch" :
    level.includes("IMPROVING") ? "risk-badge risk-improving" :
    "risk-badge risk-stable";

  // Strip the emoji prefix for display — it's redundant with the color
  const label = level.replace(/^[🔴🟡🟢]\s*/, "");
  return <span className={cls}>{label}</span>;
}

// ---------------------------------------------------------------------------
// STAT CARD — summary tile at the top of the dashboard
// ---------------------------------------------------------------------------
function StatCard({ label, value, accent }) {
  return (
    <div className={`card stat-card ${accent ? "stat-accent-" + accent : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DASHBOARD — main page
// ---------------------------------------------------------------------------
export default function Dashboard() {
  const [accounts, setAccounts] = useState([]);
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        // Fetch accounts and eval metrics concurrently
        const [accRes, evalRes] = await Promise.all([
          api.get("/accounts"),
          api.get("/eval").catch(e => {
            console.warn("Could not fetch eval data", e);
            return { data: null };
          })
        ]);
        
        setAccounts(
          accRes.data.sort((a, b) => {
            const order = (r) =>
              r.includes("HIGH") ? 0 : r.includes("WATCH") ? 1 : 2;
            return order(a.risk_level) - order(b.risk_level);
          })
        );
        
        if (evalRes.data) {
          setEvalData(evalRes.data);
        }
      } catch (e) {
        setError("Could not reach the backend. Is the FastAPI server running?");
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Derive summary counts from the loaded data
  const highRisk  = accounts.filter((a) => a.risk_level.includes("HIGH")).length;
  const watching  = accounts.filter((a) => a.risk_level.includes("WATCH")).length;
  const stable    = accounts.filter((a) =>
    a.risk_level.includes("STABLE") || a.risk_level.includes("IMPROVING")
  ).length;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">ClientPulse</span>
          <h1>Risk Dashboard</h1>
        </div>
        <nav className="nav">
          <Link to="/">Dashboard</Link>
          <Link to="/copilot">Copilot</Link>
        </nav>
      </header>

      <main className="page" style={{ maxWidth: 1200, margin: "0 auto" }}>

        {/* ── Summary stat tiles ── */}
        <section className="panel hero-panel">
          <div>
            <p className="eyebrow accent">Overview</p>
            <h2>Account health at a glance</h2>
          </div>
          {loading ? (
            <div className="stats-grid">
              <div className="card muted">Loading…</div>
            </div>
          ) : (
            <div className="stats-grid">
              <StatCard label="Total accounts"  value={accounts.length} />
              <StatCard label="High risk"       value={highRisk}  accent="danger"  />
              <StatCard label="Watching"        value={watching}  accent="warning" />
              <StatCard label="Stable / Improving" value={stable} accent="success" />
            </div>
          )}
        </section>

        {/* ── Error state ── */}
        {error && (
          <div className="panel" style={{ color: "var(--danger)" }}>{error}</div>
        )}

        {/* ── Evaluation Summary Panel (Step 5) ── */}
        {!loading && !error && evalData && (
          <section className="panel" style={{ background: "rgba(52, 211, 153, 0.05)", borderColor: "rgba(52, 211, 153, 0.2)" }}>
            <div className="section-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <h2>System Evaluation</h2>
                <span className="risk-badge risk-stable" style={{ fontSize: '0.7rem' }}>Phase 6 Harness</span>
              </div>
              <span className="muted">Live test results</span>
            </div>

            <div className="eval-grid">
              <StatCard label="Accuracy"  value={`${(evalData.accuracy * 100).toFixed(0)}%`} />
              <StatCard label="Precision" value={`${(evalData.precision * 100).toFixed(0)}%`} />
              <StatCard label="Recall"    value={`${(evalData.recall * 100).toFixed(0)}%`} />
            </div>

            {evalData.accounts && evalData.accounts.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Per-Account Correctness</h3>
                <div style={{ display: 'grid', gap: 6 }}>
                  {evalData.accounts.map(acc => (
                    <div key={acc.account_id} className="eval-row">
                      <span className="eval-account">{acc.account_id}</span>
                      <div style={{ display: 'flex', gap: 16, fontSize: '0.75rem' }}>
                        <span>Expected: <strong>{acc.expected}</strong></span>
                        <span>Actual: <strong>{acc.actual}</strong></span>
                        <span className={acc.correct ? "eval-correct" : "eval-wrong"}>
                          {acc.correct ? "✓ PASS" : "✗ FAIL"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* ── Account list with sparklines ── */}
        {!loading && !error && (
          <section className="panel">
            <div className="section-header">
              <h2>Accounts</h2>
              <span className="muted">{accounts.length} total</span>
            </div>

            <div className="account-list">
              {accounts.map((acct) => (
                <Link
                  key={acct.account_id}
                  className="account-row"
                  to={`/account/${acct.account_id}`}
                  id={`account-row-${acct.account_id}`}
                >
                  {/* Left: name + dates + message count */}
                  <div className="account-row-info">
                    <div className="account-name">{acct.account_id}</div>
                    <div className="flag-meta">
                      {acct.first_date} → {acct.last_date}
                      &nbsp;·&nbsp; {acct.n_messages} messages
                    </div>
                    {/* Show the explanation only for flagged accounts */}
                    {acct.explanation && (
                      <div className="flag-text" style={{ marginTop: 6, fontSize: "0.8rem" }}>
                        {acct.explanation}
                      </div>
                    )}
                  </div>

                  {/* Center: risk badge + trend numbers */}
                  <div className="account-row-badge">
                    <RiskBadge level={acct.risk_level} />
                    <div className="trend-meta">
                      <span title="Slope: direction of change over all messages">
                        slope {acct.slope > 0 ? "+" : ""}{acct.slope.toFixed(3)}
                      </span>
                      <span title="Drift: how far recent avg is from this account's own baseline">
                        drift {acct.drift > 0 ? "+" : ""}{acct.drift.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  {/* Right: sparkline */}
                  <div className="account-row-chart">
                    <Sparkline
                      series={acct.series}
                      baseline={acct.baseline}
                      riskLevel={acct.risk_level}
                    />
                  </div>
                </Link>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
