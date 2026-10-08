# pyrefly: ignore [missing-import]
import streamlit as st
import requests
import os

API_URL = os.environ.get("CLIENTPULSE_API", "http://localhost:8000")

st.title("ClientPulse — Account Health Dashboard")

st.sidebar.title("Controls")
acct = st.sidebar.text_input("Account ID", value="acct_techcorp")
question = st.sidebar.text_input("Copilot question", value="Why is this account flagged and what are the specific concerns?")

if st.sidebar.button("Refresh flags"):
    try:
        flags = requests.get(f"{API_URL}/flags").json()
        st.sidebar.write(flags)
    except Exception as e:
        st.sidebar.error(str(e))

st.header("Account Overview")
if acct:
    try:
        resp = requests.get(f"{API_URL}/account/{acct}")
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Trend")
            st.json(data["trend"])
            st.subheader("Recent messages")
            for m in data["recent_messages"]:
                st.write(f"{m['date'][:10]} | {m['channel']} | {m['text']}")
        else:
            st.write(f"No data for {acct}")
    except Exception as e:
        st.error(str(e))

st.header("Copilot")
if st.button("Ask Copilot"):
    try:
        resp = requests.post(f"{API_URL}/ask", json={"account_id": acct, "question": question})
        st.write(resp.json().get("answer"))
        st.subheader("Evidence")
        st.json(resp.json().get("evidence"))
    except Exception as e:
        st.error(str(e))
