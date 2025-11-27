import streamlit as st
import requests
import time
from datetime import datetime

st.set_page_config(
    page_title="CrisisGuard AI - Misinformation Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-log {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        border-left: 3px solid #667eea;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ CrisisGuard AI</h1>
    <p>Agentic AI-Powered Misinformation Detection System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Info")
    st.info("""
    **Active Agents:**
    1. 🔍 Claim Extraction
    2. 🔎 Evidence Retrieval
    3. 📄 Web Scraping
    4. ⚖️ Credibility Scoring
    5. 🧠 NLI Reasoning
    6. 📊 Synthesis
    """)
    
    st.header("📊 API Usage")
    api_calls = st.empty()
    api_calls.metric("Google API Calls Today", "0/100")
    
    st.header("ℹ️ How It Works")
    st.markdown("""
    1. **Extract Claims**: Identifies verifiable statements
    2. **Search Evidence**: Multi-source fact-checking
    3. **Score Credibility**: Evaluates source trustworthiness
    4. **Analyze Logic**: NLI-based verification
    5. **Synthesize**: Generates verdict
    """)

backend_url = "http://127.0.0.1:8000/detect"

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    mode = st.radio("Select Input Type:", ["Text", "URL"], horizontal=True)
    user_input = st.text_area(
        "Enter the text or URL to verify:",
        height=150,
        placeholder="Paste a news claim or article URL here..."
    )

with col2:
    st.markdown("### 💡 Example Claims")
    examples = [
        "Scientists discover coffee cures cancer with 100% success rate",
        "COVID-19 vaccines contain microchips for tracking",
        "The Earth is only 6000 years old"
    ]
    for i, example in enumerate(examples):
        if st.button(f"Try Example {i+1}", key=f"ex{i}"):
            user_input = example
            st.rerun()

# Analyze button
if st.button("🔍 Analyze with AI Agents", use_container_width=True, type="primary"):
    
    if not user_input.strip():
        st.warning("⚠️ Please enter text or a URL.")
        st.stop()
    
    payload = {"text": user_input} if mode == "Text" else {"url": user_input}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    agent_logs_container = st.empty()
    
    start_time = time.time()
    
    try:
        with st.spinner("🤖 Agents are analyzing..."):
            response = requests.post(backend_url, json=payload, timeout=60).json()
        
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        
    except requests.exceptions.ConnectionError:
        st.error("❌ **Backend server is not running!**")
        st.code("python main.py  # Start FastAPI backend first")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Check for errors
    if "error" in response:
        st.error(f"❌ {response['error']}")
        st.stop()
    
    # Display results
    st.markdown("---")
    st.markdown("## 🎯 Analysis Results")
    
    # Verdict section
    category = response.get("category", "Unknown")
    confidence = round(float(response.get("confidence", 0)) * 100, 2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if category == "Likely Misinformation":
            st.error(f"🔴 **{category}**")
        elif category == "Authentic Information":
            st.success(f"🟢 **{category}**")
        else:
            st.warning(f"🟡 **{category}**")
    
    with col2:
        st.metric("Confidence Score", f"{confidence}%")
    
    with col3:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    # Explanation
    st.markdown("### 📝 Claim Analyzed")
    st.info(response.get("claim", "No claim extracted"))
    
    st.markdown("### 💬 Explanation")
    st.write(response.get("explanation", "No explanation available"))
    
    # Agent Logs (Expandable)
    with st.expander("🤖 View Agent Execution Log", expanded=False):
        agent_logs = response.get("agent_logs", [])
        for log in agent_logs:
            st.markdown(f'<div class="agent-log">{log}</div>', unsafe_allow_html=True)
    
    # Sources
    st.markdown("### 🔗 Evidence Sources")
    
    verified_sources = response.get("verified_sources", [])
    
    if verified_sources:
        for i, src in enumerate(verified_sources[:5]):
            with st.container():
                st.markdown(f"**{i+1}. [{src.get('title', 'Untitled')}]({src.get('url', '#')})**")
                st.caption(src.get('snippet', 'No description available'))
                st.markdown("---")
    else:
        st.warning("No credible sources found for verification")
    
    # Metrics
    st.markdown("### 📊 System Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Google API Calls Used", response.get("google_api_calls", 0))
    
    with metric_col2:
        st.metric("Sources Analyzed", len(verified_sources))
    
    with metric_col3:
        st.metric("Claims Extracted", len(response.get("agent_logs", [])))
    
    # Update sidebar
    api_calls.metric(
        "Google API Calls Today", 
        f"{response.get('google_api_calls', 0)}/100"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by LangGraph Agentic AI | Built for Hackathon 2025</p>
    <p>Using: DuckDuckGo (Free) + Google Custom Search (100/day) + HuggingFace NLI</p>
</div>
""", unsafe_allow_html=True)