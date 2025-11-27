# ============================================================
# main3.py — AI-Ensemble Misinformation Detector
# Google CSE + Webscraping + NLI (BART MNLI)
# Gemini 2.5 Reasoning + Llama 3.1 Ensemble
# ============================================================

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime

# ============================================================
# LOAD ENVIRONMENT
# ============================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ============================================================
# FASTAPI INIT
# ============================================================
app = FastAPI(title="CrisisGuard — AI Ensemble Backend")

# ============================================================
# LOAD ML NLI MODEL
# ============================================================
print("Loading NLI model (BART-large-MNLI)...")
nli_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)
print("NLI model loaded.")

# ============================================================
# REQUEST MODELS
# ============================================================
class DetectRequest(BaseModel):
    text: str | None = None
    url: str | None = None

class Evidence(BaseModel):
    title: str | None
    url: str | None
    snippet: str | None

class DetectResponse(BaseModel):
    claim: str
    final_label: str
    trust_score: int
    ml_label: str
    gemini_label: str
    llama_label: str
    evidence: list[Evidence]
    logs: list[str]
    processing_time: float
    timestamp: str

# ============================================================
# HELPERS
# ============================================================
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extract_claim(text: str):
    text = clean_text(text)
    if len(text.split()) <= 25:
        return text
    return text.split(".")[0]

# ============================================================
# GOOGLE SEARCH
# ============================================================
def google_search(query: str):
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": 5
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()
    items = data.get("items", []) or []

    results = []
    for it in items:
        results.append({
            "title": it.get("title"),
            "url": it.get("link"),
            "snippet": it.get("snippet")
        })

    return results

# ============================================================
# WEB SCRAPER
# ============================================================
def scrape_page(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.text.strip() for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        return clean_text(text[:1200])
    except:
        return ""

# ============================================================
# BART MNLI CLASSIFIER
# ============================================================
def ml_nli_label(claim, evidence):
    if not evidence:
        return "UNCERTAIN"

    text = evidence[0]["snippet"][:300]

    result = nli_model(
        text,
        candidate_labels=["supports", "contradicts", "irrelevant"],
        hypothesis_template=f"This text {{}} the claim: '{claim}'."
    )

    label = result["labels"][0]
    score = float(result["scores"][0])

    if label == "supports" and score > 0.55:
        return "REAL"
    if label == "contradicts" and score > 0.55:
        return "MISINFORMATION"
    return "UNCERTAIN"

# ============================================================
# GEMINI REASONING
# ============================================================
def gemini_reasoning(claim, evidence):
    text = "\n".join(ev["snippet"][:400] for ev in evidence)

    prompt = f"""
CLAIM:
{claim}

EVIDENCE:
{text}

Return one word:
REAL
MISINFORMATION
UNCERTAIN
"""

    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        out = r.json()

        reply = out["candidates"][0]["content"]["parts"][0]["text"].upper()

        for w in ["REAL", "MISINFORMATION", "UNCERTAIN"]:
            if w in reply:
                return w
    except:
        pass

    return "UNCERTAIN"

# ============================================================
# LLAMA REASONING
# ============================================================
def llama_reasoning(claim, evidence):
    text = "\n".join(ev["snippet"][:350] for ev in evidence)

    prompt = f"""
CLAIM: {claim}
EVIDENCE: {text}

Return ONLY one word:
REAL
MISINFORMATION
UNCERTAIN
"""

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": "meta-llama/llama-3.1-70b-instruct",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        txt = r.json()["choices"][0]["message"]["content"].upper()

        for w in ["REAL", "MISINFORMATION", "UNCERTAIN"]:
            if w in txt:
                return w
    except:
        pass

    return "UNCERTAIN"

# ============================================================
# ENSEMBLE
# ============================================================
def ensemble(ml, gem, llama):
    votes = [ml, gem, llama]
    if votes.count("MISINFORMATION") >= 2:
        return "MISINFORMATION"
    if votes.count("REAL") >= 2:
        return "REAL"
    return "UNCERTAIN"

# ============================================================
# PIPELINE
# ============================================================
def run_pipeline(text: str, url: str, logs: list):

    start = time.time()

    # Claim Extraction
    raw = text or url or ""
    claim = extract_claim(raw)
    logs.append(f"CLAIM: {claim}")

    # Google Search
    logs.append("Searching Google...")
    results = google_search(claim)
    logs.append(f"Google results: {len(results)}")

    # Scrape
    evidence = []
    for r in results[:3]:
        page_txt = scrape_page(r["url"])
        snippet = page_txt if page_txt.strip() else r["snippet"]
        evidence.append({
            "title": r["title"],
            "url": r["url"],
            "snippet": snippet
        })

    # ML Reasoning
    ml = ml_nli_label(claim, evidence)
    gem = gemini_reasoning(claim, evidence)
    llama = llama_reasoning(claim, evidence)

    final = ensemble(ml, gem, llama)
    trust = round((sum([ml == final, gem == final, llama == final]) / 3) * 100)

    return {
        "claim": claim,
        "final_label": final,
        "trust_score": trust,
        "ml_label": ml,
        "gemini_label": gem,
        "llama_label": llama,
        "evidence": evidence,
        "logs": logs,
        "processing_time": time.time() - start,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================
# API ROUTE
# ============================================================
@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):

    logs = []

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise HTTPException(status_code=500, detail="Missing Google API credentials.")

    try:
        result = run_pipeline(req.text or "", req.url or "", logs)
        return DetectResponse(**result)
    except Exception as e:
        logs.append(str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main3:app", host="0.0.0.0", port=8000, reload=False)