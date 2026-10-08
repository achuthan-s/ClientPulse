import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import api from "../api";

// ---------------------------------------------------------------------------
// COPILOT (Step 4)
// Upgrades:
// - Dropdown for Account ID (populated from /api/accounts)
// - Suggested questions chips
// - Visually distinct states for "not enough evidence" vs grounded answers
// - Clickable citation chips in the answer text that scroll to the evidence
// ---------------------------------------------------------------------------

const SUGGESTED_QUESTIONS = [
  "Why is this account flagged?",
  "What is the client most frustrated about?",
  "Have there been any scope disputes?",
  "What was the sentiment in the last meeting?",
];

// ---------------------------------------------------------------------------
// PARSE ANSWER TEXT FOR CITATIONS
// The backend prompt specifically instructs the LLM to use citations in the
// exact format: [YYYY-MM-DD | channel]. We use regex to find these, turn
// them into clickable chips, and leave the rest as plain text.
// ---------------------------------------------------------------------------
function AnswerWithCitations({ text, onCitationClick }) {
  if (!text) return null;

  // Regex to match our specific citation format, e.g. [2023-04-14 | email]
  // We use capturing groups so we can extract the date/channel if needed,
  // but mostly we just want to split the string.
  const citationRegex = /(\[\d{4}-\d{2}-\d{2}\s*\|\s*[^\]]+\])/g;
  const parts = text.split(citationRegex);

  return (
    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
      {parts.map((part, i) => {
        // If this part is a citation bracket, render a clickable chip
        if (citationRegex.test(part)) {
          return (
            <button
              key={i}
              className="citation-chip"
              onClick={() => onCitationClick(part)}
              title="View evidence"
            >
              {part}
            </button>
          );
        }
        // Otherwise, render normal text
        return <span key={i}>{part}</span>;
      })}
    </div>
  );
}

export default function Copilot() {
  const [accounts, setAccounts] = useState([]);
  const [account, setAccount]   = useState("");
  const [question, setQuestion] = useState(SUGGESTED_QUESTIONS[0]);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);
  
  const evidenceListRef = useRef(null);

  // 1. Fetch available accounts for the dropdown
  useEffect(() => {
    async function loadAccounts() {
      try {
        const res = await api.get("/accounts");
        setAccounts(res.data);
        if (res.data.length > 0) {
          // Default to the first HIGH RISK account, or just the first one
          const highRisk = res.data.find(a => a.risk_level.includes("HIGH"));
          setAccount(highRisk ? highRisk.account_id : res.data[0].account_id);
        }
      } catch (e) {
        console.error("Failed to load accounts", e);
      }
    }
    loadAccounts();
  }, []);

  // 2. Ask the Copilot
  const ask = async (q = question) => {
    if (!account || !q) return;
    
    setQuestion(q);
    setLoading(true);
    setResult(null);
    setError(null);
    
    try {
      const res = await api.post("/ask", { account_id: account, question: q });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  // 3. Handle citation click (scroll to the referenced evidence item)
  // The citation string looks like "[2023-04-14 | email]". We find the
  // evidence item in the DOM that contains this string and scroll to it.
  const handleCitationClick = (citationStr) => {
    if (!evidenceListRef.current) return;
    
    // Simple approach: find the element containing the date
    // Extract date from the citation: "[2023-04-14 | email]" -> "2023-04-14"
    const dateMatch = citationStr.match(/\[(\d{4}-\d{2}-\d{2})/);
    if (!dateMatch) return;
    
    const targetDate = dateMatch[1];
    
    // Find all evidence items
    const items = evidenceListRef.current.querySelectorAll('.evidence-item');
    for (const item of items) {
      if (item.textContent.includes(targetDate)) {
        item.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Flash the item to highlight it
        item.style.backgroundColor = 'rgba(96, 165, 250, 0.2)';
        setTimeout(() => {
          item.style.backgroundColor = '';
        }, 1500);
        break;
      }
    }
  };

  // Determine if the answer is an "ungrounded" refusal
  const answerText = result?.answer || "";
  const isUngrounded = answerText.includes("I don't have enough evidence");

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">AI assistant</span>
          <h1>Copilot</h1>
        </div>
        <nav className="nav">
          <Link to="/">Dashboard</Link>
          <Link to="/copilot">Copilot</Link>
        </nav>
      </header>

      <main className="page" style={{ maxWidth: 1000, margin: "0 auto" }}>
        
        {/* ── QUESTION PANEL ── */}
        <section className="panel form-panel">
          <div className="field">
            <label>Select Account</label>
            <select
              value={account}
              onChange={(e) => setAccount(e.target.value)}
              disabled={accounts.length === 0}
            >
              {accounts.map(a => (
                <option key={a.account_id} value={a.account_id}>
                  {a.account_id} {a.risk_level.includes("HIGH") ? "🔴" : a.risk_level.includes("WATCH") ? "🟡" : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Question</label>
            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about this account..."
            />
            <div className="suggested-questions">
              {SUGGESTED_QUESTIONS.map(q => (
                <button 
                  key={q} 
                  className="suggested-q-btn"
                  onClick={() => ask(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          <button 
            className="primary-btn" 
            onClick={() => ask()} 
            disabled={loading || !account || !question}
            style={{ opacity: loading ? 0.7 : 1, marginTop: 8 }}
          >
            {loading ? "Analyzing Evidence..." : "Ask Copilot"}
          </button>
          
          {error && <div style={{ color: "var(--danger)", marginTop: 8 }}>{error}</div>}
        </section>

        {/* ── ANSWER PANEL ── */}
        {result && (
          <section className="panel response-panel">
            <h2>Answer</h2>
            <div className={`response-box ${isUngrounded ? 'answer-ungrounded' : 'answer-grounded'}`}>
               <AnswerWithCitations 
                 text={answerText} 
                 onCitationClick={handleCitationClick} 
               />
            </div>
            
            {result.evidence?.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h3 style={{ fontSize: '1rem', marginBottom: 12, color: 'var(--muted)' }}>
                  Retrieved Evidence ({result.evidence.length})
                </h3>
                <ul className="evidence-list" ref={evidenceListRef}>
                  {result.evidence.map((msg, i) => {
                    const sentimentColor =
                      msg.sentiment < -0.5 ? "var(--danger)"  :
                      msg.sentiment < 0    ? "var(--warning)" : "var(--muted)";
                    
                    return (
                      <li key={i} className="evidence-item" style={{ transition: 'background-color 0.5s' }}>
                        <div className="evidence-meta">
                          <span>{msg.date?.slice(0, 10)}</span>
                          <span style={{ textTransform: "capitalize" }}>{msg.channel?.replace("_", " ")}</span>
                          <span style={{ color: sentimentColor, fontWeight: 700 }}>
                            {msg.sentiment > 0 ? "+" : ""}{msg.sentiment?.toFixed(2)}
                          </span>
                        </div>
                        <p style={{ marginTop: 6 }}>{msg.text || msg.text_snippet}</p>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
