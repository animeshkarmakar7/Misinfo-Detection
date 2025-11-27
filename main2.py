# ============================================
# main2.py — Unified Agentic Misinformation Detector
# (Google CSE + Web Scraping + NLI Scoring + Verdict)
# ============================================

import os
import re
import time
import string
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime

# --------------------------------------------
# LOAD ENVIRONMENT
# --------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --------------------------------------------
# CONFIG
# --------------------------------------------
MAX_GOOGLE_RESULTS = 6
REQUEST_TIMEOUT = 12
MAX_SCRAPED_CHARS = 3000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

app = FastAPI(title="CrisisGuard — Unified Agentic Backend")


# ============================================
# MODELS
# ============================================
class DetectRequest(BaseModel):
    text: str | None = None
    url: str | None = None

class SourceOut(BaseModel):
    title: str | None
    url: str
    snippet: str | None
    scraped_text: str | None
    domain_score: float
    source_label: str
    source_score: float

class DetectResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    explanation: str
    sources: list[SourceOut]
    processing_time: float
    timestamp: str
    logs: list[str]


# ============================================
# HELPERS
# ============================================
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extract_claim(text: str) -> str:
    """Use short text as claim; else select meaningful sentence."""
    if not text:
        return ""
    text = clean_text(text)
    tokens = text.split()

    if len(tokens) <= 20:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = ""
    best_score = -1

    for s in sentences:
        score = 0
        if re.search(r"\d", s):
            score += 1
        score += min(len(s.split()) / 20.0, 1.0)
        if score > best_score:
            best_score = score
            best = s.strip()

    return best or text


def google_search(query: str):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("Google API key or CSE ID missing.")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(10, MAX_GOOGLE_RESULTS)
    }

    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    data = r.json()
    items = data.get("items", []) or []
    return [
        {
            "title": it.get("title"),
            "url": it.get("link"),
            "snippet": it.get("snippet")
        }
        for it in items
    ]


def scrape_page(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

    if r.status_code != 200 or not r.text:
        raise RuntimeError(f"Non-200 or empty response: {url}")

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "iframe", "noscript"]):
        tag.decompose()

    article = soup.find("article")
    text = ""

    if article:
        text = clean_text(article.get_text(separator=" ", strip=True))

    if not text:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([clean_text(p) for p in paragraphs if p])

    text = clean_text(text)
    return text[:MAX_SCRAPED_CHARS] + ("..." if len(text) > MAX_SCRAPED_CHARS else "")


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except:
        return ""


def score_domain(domain: str) -> float:
    TRUSTED = {
        "bbc.com": 0.92, "nytimes.com": 0.87,
        "theguardian.com": 0.88, "reuters.com": 0.95,
        "apnews.com": 0.95, "who.int": 0.95,
        "cdc.gov": 0.95, "nih.gov": 0.94
    }

    UNRELIABLE = {
        "infowars.com": 0.15,
        "beforeitsnews.com": 0.18
    }

    if domain in TRUSTED:
        return TRUSTED[domain]
    if domain in UNRELIABLE:
        return UNRELIABLE[domain]
    if domain.endswith(".gov"):
        return 0.85
    if domain.endswith(".edu"):
        return 0.75
    return 0.5


SUPPORT_HINTS = {"confirmed", "reported", "study", "evidence", "found"}
REFUTE_HINTS = {"fake", "debunked", "not true", "false", "misleading", "no evidence"}
NEGATION = {"not", "no", "never", "none"}


def nli_score(claim: str, evidence: str):
    claim = clean_text(claim)
    ev = clean_text(evidence)

    if not ev:
        return {"label": "neutral", "score": 0.0}

    c_tokens = set(re.sub(r"[^A-Za-z0-9]+", " ", claim).lower().split())
    e_tokens = set(re.sub(r"[^A-Za-z0-9]+", " ", ev).lower().split())

    overlap = len(c_tokens & e_tokens) / max(1, len(c_tokens))

    ev_lower = ev.lower()
    support_hits = any(w in ev_lower for w in SUPPORT_HINTS)
    refute_hits = any(w in ev_lower for w in REFUTE_HINTS)

    score = overlap

    if support_hits:
        return {"label": "supports", "score": max(0.6, score)}
    if refute_hits or (NEGATION & e_tokens and overlap > 0.1):
        return {"label": "refutes", "score": max(0.6, score)}

    if overlap > 0.35:
        return {"label": "supports", "score": overlap}

    return {"label": "neutral", "score": overlap}


# ============================================
# PIPELINE
# ============================================
def run_pipeline(input_text: str, input_url: str, logs: list[str]):
    start = time.time()

    # CLAIM EXTRACTION
    text_for_claim = input_text or ""
    if input_url and not text_for_claim:
        logs.append(f"Extracting text from URL for claim: {input_url}")
        try:
            text_for_claim = scrape_page(input_url)
        except Exception as e:
            logs.append(f"URL extraction failed: {e}")

    claim = extract_claim(text_for_claim)
    if not claim:
        raise RuntimeError("No claim could be extracted.")

    logs.append(f"CLAIM: {claim}")

    # GOOGLE SEARCH
    logs.append("Running Google search...")
    try:
        results = google_search(claim)
        logs.append(f"Google returned {len(results)} results.")
    except Exception as e:
        logs.append(f"Google error: {e}")
        raise RuntimeError(f"Google search failed: {e}")

    if not results:
        return {
            "claim": claim,
            "verdict": "Cannot Verify",
            "confidence": 0.0,
            "explanation": "Google returned no evidence sources.",
            "sources": [],
            "processing_time": time.time() - start,
            "logs": logs
        }

    # SCRAPING + NLI SCORING
    source_objs = []
    support_weight = refute_weight = total_weight = 0

    for idx, it in enumerate(results[:MAX_GOOGLE_RESULTS]):
        url = it["url"]
        title = it["title"]
        snippet = it["snippet"]

        logs.append(f"Scraping {idx+1}: {url}")

        scraped = ""
        try:
            scraped = scrape_page(url)
            logs.append(f"Scraped {len(scraped)} chars.")
        except Exception as e:
            logs.append(f"Scrape failed: {e}")
            scraped = snippet or ""

        domain = domain_from_url(url)
        dscore = score_domain(domain)

        nli = nli_score(claim, scraped)

        source_objs.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "scraped_text": scraped,
            "domain_score": dscore,
            "source_label": nli["label"],
            "source_score": nli["score"]
        })

        total_weight += dscore
        if nli["label"] == "supports":
            support_weight += nli["score"] * dscore
        elif nli["label"] == "refutes":
            refute_weight += nli["score"] * dscore

    # VERDICT
    if total_weight == 0:
        verdict = "Cannot Verify"
        confidence = 0.0
    else:
        s_norm = support_weight / total_weight
        r_norm = refute_weight / total_weight

        if s_norm >= 0.55 and s_norm > r_norm:
            verdict = "Supported"
            confidence = s_norm
        elif r_norm >= 0.55 and r_norm > s_norm:
            verdict = "Refuted"
            confidence = r_norm
        else:
            verdict = "Needs More Evidence"
            confidence = max(s_norm, r_norm)

    # Explanation
    if verdict == "Supported":
        explanation = "Evidence strongly supports the claim."
    elif verdict == "Refuted":
        explanation = "Evidence contradicts the claim."
    else:
        explanation = "Evidence is mixed or insufficient."

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "explanation": explanation,
        "sources": source_objs,
        "processing_time": time.time() - start,
        "logs": logs
    }


# ============================================
# ENDPOINT
# ============================================
@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):

    logs = []

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise HTTPException(status_code=500, detail="Google API key / CSE ID not configured.")

    if not req.text and not req.url:
        raise HTTPException(status_code=400, detail="Provide text or URL.")

    try:
        text = req.text or ""
        url = req.url or ""

        result = run_pipeline(text, url, logs)

        return DetectResponse(
            claim=result["claim"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            sources=[SourceOut(**s) for s in result["sources"]],
            processing_time=result["processing_time"],
            timestamp=datetime.utcnow().isoformat(),
            logs=result["logs"]
        )

    except Exception as e:
        logs.append(str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)