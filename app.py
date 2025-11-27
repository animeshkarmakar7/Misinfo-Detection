import streamlit as st
import requests

st.set_page_config(page_title="CrisisGuard – Misinformation Detector", layout="wide")

st.title("🛡️ CrisisGuard – Real-Time Misinformation Detector")
st.write("Analyze news, claims, and URLs with cross-checking from trusted sources.")

backend_url = "http://127.0.0.1:8000/detect"

mode = st.radio("Select Input Type:", ["Text", "URL"])

user_input = st.text_area("Enter the text or URL to verify:", height=120)

if st.button("Analyze", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter text or a URL.")
    else:
        payload = {"text": user_input} if mode == "Text" else {"url": user_input}

        with st.spinner("Analyzing and fact-checking..."):
            try:
                response = requests.post(backend_url, json=payload).json()
            except:
                st.error("❌ Backend server is not running. Start FastAPI first.")
                st.stop()

        # Handle backend errors safely
        if "error" in response:
            st.error(response["error"])
            st.stop()

        category = response["category"]
        confidence = round(float(response["confidence"]) * 100, 2)

        # verdict box
        if category == "Likely Misinformation":
            st.error(f"🔴 {category} — {confidence}% confidence")
        elif category == "Authentic Information":
            st.success(f"🟢 {category} — {confidence}% confidence")
        else:
            st.warning(f"🟡 {category} — {confidence}% confidence")

        # Claim Text
        st.markdown("### 📝 Claim Analyzed:")
        st.write(response["claim"])

        # Evidence Snippet (SAFE)
        st.markdown("### 🔍 Evidence Snippet from Trusted Source:")
        if response.get("matched_evidence"):
            st.info(response["matched_evidence"]["snippet"])
        else:
            st.warning("No supporting evidence found.")

        # Sources List
        st.markdown("### 🔗 Verified Trusted Sources:")
        if response.get("verified_sources"):
            for src in response["verified_sources"]:
                st.markdown(
                    f"""
                    **🔹 [{src['title']}]({src['url']})**  
                    *{src['snippet']}*
                    """
                )
        else:
            st.info("No credible sources found.")
