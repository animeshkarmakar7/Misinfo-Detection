import streamlit as st
import requests
import time

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="🛰️ FactRadar – Fast Misinformation Detector",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/detect"

st.title("🛰️ FactRadar — Hybrid Misinformation Detection System")
st.markdown("### 🔍 Claim Extraction → Google Search → Web Scraping → Credibility Scoring")
st.markdown("---")

# =========================================================
# INPUT
# =========================================================
mode = st.radio("Choose input type:", ["Claim Text", "URL"], horizontal=True)

user_input = st.text_area(
    "Paste a claim or URL:",
    height=140,
    placeholder="Example: Scientists discovered a new COVID variant in 2020 that spreads twice as fast globally."
)

# =========================================================
# ANALYZE BUTTON
# =========================================================
if st.button("Analyze", use_container_width=True):

    st.markdown("⏳ **Running analysis...**")
    start = time.time()

    # Prepare request payload
    payload = {
        "text": user_input if mode == "Claim Text" else None,
        "url": user_input if mode == "URL" else None,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=180)
        resp.raise_for_status()
        result = resp.json()
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

    verdict = result.get("verdict", "Unknown")
    conf = result.get("confidence", 0.0)

    if verdict == "Supported":
        st.success(f"🟢 Supported — Confidence {conf * 100:.1f}%")
    elif verdict == "Refuted":
        st.error(f"🔴 Refuted — Confidence {conf * 100:.1f}%")
    elif verdict == "Needs More Evidence":
        st.warning(f"🟡 Needs More Evidence — Confidence {conf * 100:.1f}%")
    else:
        st.info(f"⚪ Cannot Verify — Confidence {conf * 100:.1f}%")

    st.markdown("---")

    # =========================================================
    # CLAIM
    # =========================================================
    st.subheader("🗣️ Extracted Claim")
    st.info(result.get("claim", "No claim extracted"))

    st.markdown("---")

    # =========================================================
    # EXPLANATION
    # =========================================================
    st.subheader("📘 Explanation")
    st.write(result.get("explanation", ""))

    st.markdown("---")

    # =========================================================
    # EVIDENCE & SOURCES
    # =========================================================
    st.subheader("🔎 Evidence & Sources (Google Search + Scraping)")

    sources = result.get("sources", [])

    if not sources:
        st.warning("⚠️ No evidence sources found.")
    else:
        for idx, src in enumerate(sources):
            st.markdown(f"### Source {idx + 1}")

            st.markdown(f"**🔗 [{src['title']}]({src['url']})**")
            st.write(f"**Snippet:** {src['snippet']}")
            st.write(f"**Domain Credibility Score:** {src['domain_score']:.2f}")
            st.write(f"**Source Label:** {src['source_label'].capitalize()} ({src['source_score']:.2f})")

            with st.expander("📄 Full Extracted Text"):
                st.write(src.get("scraped_text"))

            st.markdown("---")

    # =========================================================
    # LOGS
    # =========================================================
    st.subheader("🧠 Agent Logs")
    logs = result.get("logs", [])
    with st.expander("Show logs"):
        st.code("\n".join(logs))
