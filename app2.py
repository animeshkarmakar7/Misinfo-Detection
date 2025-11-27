import streamlit as st
import requests
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🛰️ FactRadar — AI Misinformation Detector",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/detect"

st.title("🛰️ FactRadar — AI-Ensemble Misinformation Detection")
st.markdown("### ML (BART-MNLI) + Gemini 2.5 + Llama 3.1 + Web Scraping + Google CSE")
st.markdown("---")

# =========================================================
# INPUT SECTION
# =========================================================
mode = st.radio("Choose input type:", ["Claim Text", "URL"], horizontal=True)

user_input = st.text_area(
    "Paste a claim or news article URL:",
    height=140,
    placeholder="Example (Claim): COVID vaccine causes infertility\n\nExample (URL): https://example.com/news/abc"
)

# =========================================================
# ANALYZE BUTTON
# =========================================================
if st.button("Analyze", use_container_width=True):

    st.markdown("⏳ **Running AI ensemble analysis...**")

    start = time.time()

    payload = {
        "text": user_input if mode == "Claim Text" else None,
        "url": user_input if mode == "URL" else None,
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=200)
        r.raise_for_status()
        res = r.json()
    except Exception as e:
        st.error(f"❌ Backend Error: {e}")
        st.stop()

    end = time.time()
    st.success(f"Analysis completed in **{round(end - start, 2)} seconds**")
    st.markdown("---")

    # =========================================================
    # FINAL VERDICT
    # =========================================================
    st.subheader("🧭 Final Verdict")

    verdict = res["final_label"]
    trust = res["trust_score"]

    if verdict == "REAL":
        st.success(f"🟢 REAL — Trust Score: {trust}%")
    elif verdict == "MISINFORMATION":
        st.error(f"🔴 MISINFORMATION — Trust Score: {trust}%")
    else:
        st.warning(f"🟡 UNCERTAIN — Trust Score: {trust}%")

    st.markdown("---")

    # =========================================================
    # MODEL VOTES
    # =========================================================
    st.subheader("🗳️ Model Votes (AI Ensemble)")

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Model (BART-MNLI)", res["ml_label"])
    col2.metric("Gemini 2.5 Flash", res["gemini_label"])
    col3.metric("Llama 3.1 (OpenRouter)", res["llama_label"])

    st.markdown("---")

    # =========================================================
    # EXTRACTED CLAIM
    # =========================================================
    st.subheader("📝 Extracted Claim")
    st.info(res["claim"])

    st.markdown("---")

    # =========================================================
    # EVIDENCE (SCRAPED)
    # =========================================================
    st.subheader("📚 Evidence Retrieved (Google CSE + Scraping)")

    evidence = res["evidence"]

    if not evidence:
        st.warning("⚠️ No evidence sources retrieved.")
    else:
        for idx, ev in enumerate(evidence):
            st.markdown(f"### 🔗 Source {idx + 1}: [{ev['title']}]({ev['url']})")

            st.write(f"**URL:** {ev['url']}")
            st.write(f"**Extracted Text / Snippet:**")
            st.info(ev["snippet"])

            st.markdown("---")

    # =========================================================
    # LOGS
    # =========================================================
    st.subheader("🧠 Agent Logs")
    logs = res.get("logs", [])

    with st.expander("Show system logs"):
        st.code("\n".join(logs))
