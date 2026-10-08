import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  Legend,
} from "recharts";
import api from "../api";

// ---------------------------------------------------------------------------
// PURE MATH — mirrors trend_tracker.py exactly so the slider recomputes
// in the browser without a network round-trip on every drag event.
// WHY CLIENT-SIDE: With only 15 data points per account, doing this in JS
// is instant (<1ms). A server round-trip per drag frame would feel laggy.
// ---------------------------------------------------------------------------

function rollingAvg(sentiments, window = 4) {
  // Replicates trend_tracker.rolling_average()
  return sentiments.map((_, i) => {
    const slice = sentiments.slice(Math.max(0, i - window + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function computeSlope(sentiments) {
  // Replicates trend_tracker.compute_slope() — least-squares linear regression
  const n = sentiments.length;
  if (n < 2) return 0;
  const xMean = (n - 1) / 2;
  const yMean = sentiments.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  sentiments.forEach((y, i) => {
    num += (i - xMean) * (y - yMean);
    den += (i - xMean) ** 2;
  });
  return den === 0 ? 0 : num / den;
}

function classifyRisk(slope, drift) {
  // Replicates trend_tracker.classify_risk()
  if (slope < -0.03 && drift < -0.2)       return "HIGH RISK";
  if (slope < -0.01 || drift < -0.1)        return "WATCH";
  if (slope >  0.01 && drift >  0.05)       return "IMPROVING";
  return "STABLE";
}

function computeTrend(rawSeries, window = 4) {
  // rawSeries: [{date, raw, smoothed}] — we recompute smoothed ourselves
  // so it correctly uses only the visible subset (not the full series).
  if (!rawSeries.length) return null;
  const sentiments = rawSeries.map((p) => p.raw);
  const smoothed   = rollingAvg(sentiments, window);

  const half      = Math.max(1, Math.floor(sentiments.length / 2));
  const baseline  = sentiments.slice(0, half).reduce((a, b) => a + b, 0) / half;
  const recentAvg = smoothed[smoothed.length - 1];
  const drift     = recentAvg - baseline;
  const slope     = computeSlope(sentiments);
  const risk      = classifyRisk(slope, drift);

  // Rebuild series with freshly recomputed smoothed values for the subset
  const series = rawSeries.map((pt, i) => ({
    date:     pt.date,
    raw:      pt.raw,
    smoothed: +smoothed[i].toFixed(3),
  }));

  // First index where smoothed crosses below the baseline
  let declineIdx = null;
  for (let i = 1; i < series.length; i++) {
    if (series[i].smoothed < baseline) { declineIdx = i; break; }
  }

  return {
    slope:       +slope.toFixed(4),
    drift:       +drift.toFixed(3),
    baseline:    +baseline.toFixed(3),
    recent_avg:  +recentAvg.toFixed(3),
    risk,
    series,
    declineIdx,
  };
}

// ---------------------------------------------------------------------------
// PRE-COMPUTE the first index at which the account becomes flagged.
// We do this once when data loads so the slider can show the "⚡ First flagged"
// label at exactly the right moment.
// ---------------------------------------------------------------------------
function findFirstFlaggedIndex(fullSeries) {
  for (let i = 1; i < fullSeries.length; i++) {
    const subset = fullSeries.slice(0, i + 1);
    const t = computeTrend(subset);
    if (t && (t.risk === "HIGH RISK" || t.risk === "WATCH")) return i;
  }
  return null;
}

// ---------------------------------------------------------------------------
// SHARED COMPONENTS
// ---------------------------------------------------------------------------

function RiskBadge({ level }) {
  const cls =
    level === "HIGH RISK" ? "risk-badge risk-high"      :
    level === "WATCH"     ? "risk-badge risk-watch"     :
    level === "IMPROVING" ? "risk-badge risk-improving" :
                            "risk-badge risk-stable";
  return <span className={cls}>{level}</span>;
}

const TOPIC_TO_PHRASES = {
  scope_dispute:  ["scope", "scope dispute", "out of scope", "additional work"],
  payment:        ["payment", "invoice", "billing", "overdue", "refund"],
  timeline:       ["delay", "late", "deadline", "behind schedule"],
  outage:         ["outage", "down", "unavailable", "incident", "degraded"],
  cancellation:   ["cancel", "terminate", "churn", "end contract"],
  sla:            ["SLA", "sla", "breach", "penalty", "violation"],
  general_update: [],
};

const RISK_KEYWORDS = [
  "cancel", "terminate", "SLA", "breach", "penalty", "scope",
  "delay", "late", "outage", "overdue", "invoice", "refund",
  "unhappy", "frustrated", "disappointed", "concerned", "urgent",
];

function HighlightedText({ text = "", topics = [] }) {
  const phrases = useMemo(() => {
    const set = new Set(RISK_KEYWORDS.map((k) => k.toLowerCase()));
    for (const t of topics) {
      for (const p of (TOPIC_TO_PHRASES[t] || [])) set.add(p.toLowerCase());
    }
    return [...set].sort((a, b) => b.length - a.length);
  }, [topics]);

  if (!phrases.length) return <>{text}</>;

  const pattern = phrases.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const regex   = new RegExp(`(${pattern})`, "gi");
  const parts   = text.split(regex);

  return (
    <>
      {parts.map((part, i) =>
        regex.test(part)
          ? <mark key={i} className="key-phrase">{part}</mark>
          : part
      )}
    </>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const raw      = payload.find((p) => p.dataKey === "raw");
  const smoothed = payload.find((p) => p.dataKey === "smoothed");
  return (
    <div style={{
      background: "var(--bg-soft)", border: "1px solid var(--line)",
      borderRadius: 10, padding: "10px 14px", fontSize: 12, color: "var(--text)",
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      {raw      && <div>Raw sentiment: <b>{raw.value?.toFixed(3)}</b></div>}
      {smoothed && <div>Smoothed (trend): <b>{smoothed.value?.toFixed(3)}</b></div>}
    </div>
  );
}

function EvidenceCard({ msg, index }) {
  const sentimentColor =
    msg.sentiment < -0.5 ? "var(--danger)"  :
    msg.sentiment < 0    ? "var(--warning)" : "var(--muted)";
  return (
    <li
      className="evidence-item"
      id={`evidence-${msg.message_id || index}`}
      style={{ animationDelay: `${index * 50}ms`, animation: "fadeIn 0.3s ease both" }}
    >
      <div className="evidence-meta">
        <span>{msg.date?.slice(0, 10)}</span>
        <span style={{ textTransform: "capitalize" }}>{msg.channel?.replace("_", " ")}</span>
        <span style={{ color: sentimentColor, fontWeight: 700 }}>
          {msg.sentiment > 0 ? "+" : ""}{msg.sentiment?.toFixed(2)}
        </span>
        {msg.scope_dispute_signal && <span className="scope-badge">⚠ scope dispute</span>}
        {(msg.topics || []).filter(Boolean).map((t) => (
          <span key={t} style={{
            background: "var(--primary-soft)", color: "var(--primary)",
            borderRadius: 6, padding: "1px 7px", fontSize: "0.68rem", fontWeight: 600,
          }}>
            {t.replace("_", " ")}
          </span>
        ))}
      </div>
      <p>
        <HighlightedText text={msg.text || msg.text_snippet || ""} topics={msg.topics} />
      </p>
      {msg.reason && msg.reason !== "No scope issues detected in this message." && (
        <div className="evidence-reason">💡 {msg.reason}</div>
      )}
    </li>
  );
}

// ---------------------------------------------------------------------------
// TIME MACHINE SLIDER COMPONENT
// ---------------------------------------------------------------------------
function TimeMachineSlider({ fullSeries, sliderIndex, onChange, firstFlaggedIndex }) {
  const currentDate = fullSeries[sliderIndex]?.date ?? "—";
  const isFlagged   = firstFlaggedIndex != null && sliderIndex >= firstFlaggedIndex;
  const isFirstFlagFrame = sliderIndex === firstFlaggedIndex;

  return (
    <div className="time-machine-bar">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <label>⏱ Time Machine — what would the system know as of…</label>
        <span className="time-machine-date">{currentDate}</span>
      </div>

      <input
        id="time-machine-slider"
        type="range"
        min={0}
        max={fullSeries.length - 1}
        value={sliderIndex}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Scrub through message history"
      />

      {/* Tick labels at first and last date */}
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--muted)", marginTop: -4 }}>
        <span>{fullSeries[0]?.date}</span>
        <span style={{ color: "var(--muted)", fontSize: 10 }}>
          {sliderIndex + 1} of {fullSeries.length} messages visible
        </span>
        <span>{fullSeries[fullSeries.length - 1]?.date}</span>
      </div>

      {/* "⚡ First flagged" label — appears exactly when the slider crosses
          the detection point and stays visible after that */}
      {isFlagged && firstFlaggedIndex != null && (
        <div className="flag-moment-label">
          ⚡ {isFirstFlagFrame ? "Account first flagged here" : `Flagged since ${fullSeries[firstFlaggedIndex]?.date}`}
          &nbsp;·&nbsp; {fullSeries.length - 1 - firstFlaggedIndex} message(s) early warning
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MAIN COMPONENT
// ---------------------------------------------------------------------------
export default function AccountDetail() {
  const { id } = useParams();
  const [data, setData]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [sliderIndex, setSlider]  = useState(0); // will be set to max on load
  
  // Draft state
  const [draft, setDraft]               = useState(null);
  const [loadingDraft, setLoadingDraft] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setDraft(null);
      try {
        const res = await api.get(`/account/${id}`);
        setData(res.data);
        // Default slider to the last (most recent) message — "now"
        setSlider((res.data.trend?.series?.length ?? 1) - 1);
        
        // Kick off draft fetch independently so it doesn't block the page load
        setLoadingDraft(true);
        api.post(`/account/${id}/draft`)
          .then(draftRes => {
            if (draftRes.data && draftRes.data.draft) {
              setDraft(draftRes.data.draft);
            }
          })
          .catch(err => console.error("Failed to fetch draft:", err))
          .finally(() => setLoadingDraft(false));
          
      } catch (e) {
        setError(e.response?.data?.detail || e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  // ── Derived: full series from backend ────────────────────────────────────
  const fullSeries = data?.trend?.series ?? [];

  // Pre-compute first-flagged index ONCE when data loads (not on every render)
  const firstFlaggedIndex = useMemo(
    () => (fullSeries.length ? findFirstFlaggedIndex(fullSeries) : null),
    [fullSeries]
  );

  // ── Time-machine recomputation ────────────────────────────────────────────
  // Slice to the slider position and recompute all trend metrics in JS.
  // This runs on every slider move but is O(n) with n≤15 — completely instant.
  const visibleRaw = fullSeries.slice(0, sliderIndex + 1);
  const liveTrend  = useMemo(() => computeTrend(visibleRaw), [sliderIndex, fullSeries]);

  const visibleSeries  = liveTrend?.series ?? [];
  const liveBaseline   = liveTrend?.baseline ?? 0;
  const liveDeclineIdx = liveTrend?.declineIdx;
  const liveDeclinePt  = liveDeclineIdx != null ? visibleSeries[liveDeclineIdx] : null;
  const liveRisk       = liveTrend?.risk ?? "STABLE";

  // Build enriched message lookup (message_id → full message with topics+reason)
  const allMsgMap = useMemo(() => {
    const m = {};
    for (const msg of (data?.all_messages ?? [])) m[msg.message_id] = msg;
    return m;
  }, [data]);

  // Evidence visible in the slider window = messages with date ≤ slider date
  const sliderDate     = fullSeries[sliderIndex]?.date ?? "";
  const visibleMessages = useMemo(
    () => (data?.all_messages ?? []).filter((m) => m.date.slice(0, 10) <= sliderDate),
    [data, sliderDate]
  );

  // Flagged evidence = stored evidence snippets that fall within the slider window
  const rawEvidence = useMemo(() => {
    return (data?.flagged_evidence ?? [])
      .map((ev) => ({ ...ev, ...(allMsgMap[ev.message_id] ?? {}) }))
      .filter((ev) => (ev.date ?? "").slice(0, 10) <= sliderDate);
  }, [data, allMsgMap, sliderDate]);

  const displayEvidence = rawEvidence.length > 0
    ? rawEvidence
    : visibleMessages.filter((m) => m.sentiment < 0);

  const handleSlider = useCallback((val) => setSlider(val), []);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">Account review</span>
          <h1>{id}</h1>
        </div>
        <nav className="nav">
          <Link to="/">← Dashboard</Link>
          <Link to="/copilot">Copilot</Link>
        </nav>
      </header>

      <main className="page" style={{ maxWidth: 1200, margin: "0 auto" }}>

        {loading && <div className="panel muted">Loading account data…</div>}
        {error   && <div className="panel" style={{ color: "var(--danger)" }}>{error}</div>}

        {data && (
          <>
            {/* ── Risk summary — updates live as slider moves ── */}
            <section className="panel hero-panel">
              <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
                <div>
                  <p className="eyebrow accent">Trend summary</p>
                  <h2 style={{ marginTop: 2 }}>{id}</h2>
                </div>
                {/* Risk badge reflects the slider-recomputed risk, NOT the stored flag */}
                <RiskBadge level={liveRisk} />
              </div>

              <div className="stats-grid">
                <div className="card stat-card">
                  <span>Own baseline</span>
                  <strong style={{ color: "var(--primary)" }}>
                    {liveBaseline > 0 ? "+" : ""}{liveBaseline}
                  </strong>
                </div>
                <div className="card stat-card">
                  <span>Recent avg</span>
                  <strong style={{
                    color: liveTrend?.recent_avg < liveBaseline ? "var(--danger)" : "var(--success)",
                    transition: "color 0.3s ease",
                  }}>
                    {liveTrend?.recent_avg > 0 ? "+" : ""}{liveTrend?.recent_avg}
                  </strong>
                </div>
                <div className="card stat-card">
                  <span>Drift</span>
                  <strong style={{
                    color: liveTrend?.drift < -0.15 ? "var(--danger)" : "var(--muted)",
                    transition: "color 0.3s ease",
                  }}>
                    {liveTrend?.drift > 0 ? "+" : ""}{liveTrend?.drift}
                  </strong>
                </div>
                <div className="card stat-card">
                  <span>Slope</span>
                  <strong style={{
                    color: liveTrend?.slope < -0.02 ? "var(--danger)" : "var(--muted)",
                    transition: "color 0.3s ease",
                  }}>
                    {liveTrend?.slope > 0 ? "+" : ""}{liveTrend?.slope}
                  </strong>
                </div>
                <div className="card stat-card">
                  <span>Visible messages</span>
                  <strong>{visibleSeries.length} / {fullSeries.length}</strong>
                </div>
              </div>
            </section>

            {/* ── TIME MACHINE SLIDER ── */}
            {fullSeries.length > 1 && (
              <TimeMachineSlider
                fullSeries={fullSeries}
                sliderIndex={sliderIndex}
                onChange={handleSlider}
                firstFlaggedIndex={firstFlaggedIndex}
              />
            )}

            {/* ── Timeline chart — redraws to visible subset on every drag ── */}
            <section className="panel">
              <div className="section-header">
                <h2>Sentiment timeline</h2>
                <span className="muted">
                  {liveDeclinePt
                    ? `Decline below baseline from ${liveDeclinePt.date}`
                    : "No decline below baseline yet"}
                </span>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={visibleSeries}
                    margin={{ top: 16, right: 24, bottom: 8, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11, fill: "var(--muted)" }}
                      tickLine={false}
                      interval={Math.max(0, Math.floor(visibleSeries.length / 6) - 1)}
                    />
                    <YAxis
                      domain={[-1.1, 1.1]}
                      tick={{ fontSize: 11, fill: "var(--muted)" }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => v.toFixed(1)}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend
                      iconType="plainline"
                      wrapperStyle={{ fontSize: 12, color: "var(--muted)" }}
                    />

                    {/* Account's own baseline — recomputed from the visible subset.
                        As the slider moves and more data is added, the baseline shifts
                        because it's always the first-half average of visible messages. */}
                    <ReferenceLine
                      y={liveBaseline}
                      stroke="rgba(96,165,250,0.5)"
                      strokeDasharray="5 3"
                      label={{
                        value: `Baseline ${liveBaseline > 0 ? "+" : ""}${liveBaseline}`,
                        position: "insideTopRight",
                        fill: "var(--primary)",
                        fontSize: 11,
                        fontWeight: 600,
                      }}
                    />

                    <ReferenceLine
                      y={0}
                      stroke="rgba(148,163,184,0.2)"
                      strokeDasharray="2 4"
                    />

                    {/* Raw per-message sentiment dots */}
                    <Line
                      type="monotone"
                      dataKey="raw"
                      name="Raw sentiment"
                      stroke="rgba(148,163,184,0.4)"
                      strokeWidth={0}
                      dot={{ r: 3.5, fill: "rgba(148,163,184,0.55)", strokeWidth: 0 }}
                      isAnimationActive={false}
                    />

                    {/* Smoothed trend line — the main signal.
                        isAnimationActive=false so it redraws instantly on slider drag
                        rather than re-running the 800ms enter animation each time. */}
                    <Line
                      type="monotone"
                      dataKey="smoothed"
                      name="Smoothed trend"
                      stroke="var(--primary)"
                      strokeWidth={2.5}
                      dot={false}
                      isAnimationActive={false}
                    />

                    {/* Decline-start dot — only shown once the slider has enough data */}
                    {liveDeclinePt && (
                      <ReferenceDot
                        x={liveDeclinePt.date}
                        y={liveDeclinePt.smoothed}
                        r={8}
                        fill="var(--danger)"
                        stroke="#fff"
                        strokeWidth={2}
                        label={{
                          value: "⬇ Decline start",
                          position: "top",
                          fill: "var(--danger)",
                          fontSize: 11,
                          fontWeight: 700,
                        }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* ── Evidence cards — filtered to slider window ── */}
            <section className="panel">
              <div className="section-header">
                <h2>Risk evidence</h2>
                <span className="muted">
                  {displayEvidence.length} flagged message{displayEvidence.length !== 1 ? "s" : ""}
                  {" "}as of {sliderDate}
                  {data.flag?.explanation && (
                    <span style={{ marginLeft: 10, color: "var(--warning)", fontWeight: 600 }}>
                      · {data.flag.explanation}
                    </span>
                  )}
                </span>
              </div>

              {displayEvidence.length === 0 ? (
                <p className="muted" style={{ padding: "8px 0" }}>
                  No flagged evidence before this date.
                </p>
              ) : (
                <ul className="evidence-list">
                  {displayEvidence.map((msg, i) => (
                    <EvidenceCard key={msg.message_id || i} msg={msg} index={i} />
                  ))}
                </ul>
              )}
            </section>

            {/* ── Full message history — not time-machine filtered, for context ── */}
            <section className="panel">
              <div className="section-header">
                <h2>Full message history</h2>
                <span className="muted">{data.all_messages?.length} messages (all time)</span>
              </div>
              <ul className="evidence-list">
                {(data.all_messages ?? []).map((msg, i) => (
                  <EvidenceCard key={msg.message_id || i} msg={msg} index={i} />
                ))}
              </ul>
            </section>

            {/* ── SUGGESTED ACTION (Step 6) ── */}
            <section className="panel">
              <div className="section-header">
                <h2>Suggested Action</h2>
                <span className="muted">AI-generated draft</span>
              </div>
              
              {loadingDraft ? (
                <div className="draft-box muted" style={{ opacity: 0.7 }}>
                  Generating suggested action...
                </div>
              ) : draft ? (
                <div className="draft-box">
                  {draft}
                </div>
              ) : (
                <p className="muted">No suggested action available for this account.</p>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
