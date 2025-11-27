import streamlit as st
import requests
import json
from datetime import datetime

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

API_ENDPOINT = "http://127.0.0.1:8000/detect"
HEALTH_ENDPOINT = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="CrisisGuard – Misinformation Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CrisisGuard – Misinformation Detection System")
st.write("Analyze a claim, search credible sources, scrape evidence, and determine its truthfulness.")


# -------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------

def check_backend():
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None


backend_status = check_backend()
if not backend_status:
    st.error("❌ Backend is not reachable. Start using:  `python main.py`")
else:
    if backend_status.get("status") != "healthy":
        st.warning("⚠ Backend running, but Google API credentials may be missing.")
    else:
        st.success("Backend connected successfully ✔")


# -------------------------------------------------------
# INPUT SECTION
# -------------------------------------------------------

mode = st.radio("Select Input Type:", ["Text", "URL"], horizontal=True)

default_text = ""

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

user_input = st.text_area(
    "Enter the claim or URL:",
    value=st.session_state.input_value,
    height=140,
    placeholder="Example: Scientists discovered a new COVID variant in 2020 that spreads twice as fast globally."
)


# Example buttons
examples = [
    "Scientists discovered a new COVID variant in 2020 that spreads twice as fast globally.",
    "WHO confirmed that drinking hot water can cure COVID-19 infections instantly.",
    "The 2020 Australian bushfires were intentionally started by government satellites."
]

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(f"Example {i+1}"):
        st.session_state.input_value = ex
        st.experimental_rerun()


# -------------------------------------------------------
# RUN ANALYSIS
# -------------------------------------------------------

if st.button("🔍 Analyze", use_container_width=True):

    if not user_input.strip():
        st.error("Please enter text or a URL.")
        st.stop()

    # Prepare request payload
    payload = {
        "text": user_input if mode == "Text" else None,
        "url": user_input if mode == "URL" else None
    }

    try:
        with st.spinner("🤖 Running agentic verification process..."):
            r = requests.post(API_ENDPOINT, json=payload, timeout=180)
            r.raise_for_status()
            result = r.json()
    except requests.exceptions.HTTPError as err:
        st.error(f"❌ Backend error: {err}")
        st.code(r.text)
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to connect to backend: {e}")
        st.stop()


    # -------------------------------------------------------
    # DISPLAY OUTPUT
    # -------------------------------------------------------

    st.success("Analysis Complete ✔")

    # ---- Verdict ----
    st.subheader("🧾 Final Verdict")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Verdict", result.get("verdict", "Unknown"))
    with c2:
        st.metric("Confidence", f"{result.get('confidence', 0.0) * 100:.1f}%")

    st.write("---")

    # ---- Claim ----
    st.subheader("🗣️ Extracted Claim")
    st.info(result.get("claim", "No claim extracted."))

    st.write("---")

    # ---- Explanation ----
    st.subheader("📘 Explanation")
    st.write(result.get("explanation", ""))

    st.write("---")

    # ---- Evidence ----
    st.subheader("🔎 Evidence & Sources")

    sources = result.get("sources", [])
    if not sources:
        st.warning("No evidence sources found.")
    else:
        for idx, src in enumerate(sources):
            st.markdown(f"### Source #{idx+1}")

            st.write(f"**Title:** {src.get('title')}")
            st.write(f"🌐 **URL:** {src.get('url')}")
            st.write(f"📝 **Snippet:** {src.get('snippet')}")
            st.write(f"🏛 **Domain Credibility Score:** {src.get('domain_score'):.2f}")
            st.write(f"🧠 **Source Judgement:** {src.get('source_label').capitalize()} "
                     f"({src.get('source_score'):.2f})")

            with st.expander("📄 Full Extracted Text"):
                st.write(src.get("scraped_text"))

            st.write("---")

    # ---- Logs ----
    st.subheader("🧠 Agent Logs (debugging)")

    with st.expander("Show Logs"):
        logs = result.get("logs", [])
        st.code("\n".join(logs))

    # ---- Processing Time ----
    st.caption(f"⏱️ Processing time: {result.get('processing_time', 0.0):.2f} seconds")
    st.caption(f"🕒 Timestamp: {result.get('timestamp', '')}")

# streamlit run D:\MisInfo\app.py