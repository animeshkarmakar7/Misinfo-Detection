"""
main.py - Simple agentic backend (Google-only, lightweight reasoning)

Requirements:
    pip install fastapi uvicorn requests beautifulsoup4 python-dotenv

.env:
    GOOGLE_API_KEY=your_api_key_here
    GOOGLE_CSE_ID=your_cse_id_here
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests
import os
import re
import time
from datetime import datetime

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# -------------------------
# Config
# -------------------------
MAX_GOOGLE_RESULTS = 6  # up to 10 allowed; keep modest to save quota
REQUEST_TIMEOUT = 12  # seconds for web requests
MAX_SCRAPED_CHARS = 3000  # limit text returned per source
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="CrisisGuard - Google-only Misinformation Backend")

# -------------------------
# Pydantic models
# -------------------------
class DetectRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

class SourceOut(BaseModel):
    title: Optional[str]
    url: str
    snippet: Optional[str]
    scraped_text: Optional[str]
    domain_score: float
    source_label: str  # supports/refutes/neutral
    source_score: float  # 0.0 - 1.0


class DetectResponse(BaseModel):
    claim: str
    verdict: str  # Supported / Refuted / Needs More Evidence / Cannot Verify
    confidence: float  # 0.0 - 1.0
    explanation: str
    sources: List[SourceOut]
    processing_time: float
    timestamp: str
    logs: List[str]


# -------------------------
# Utility functions
# -------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    text = re.sub(r'\s+', ' ', s).strip()
    return text

def domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""

def score_domain(domain: str) -> float:
    """Simple domain reputation scoring (0.0 - 1.0). Expand lists for your use case."""
    TRUSTED = {
        "reuters.com": 0.95, "apnews.com": 0.95, "bbc.com": 0.92,
        "theguardian.com": 0.88, "nytimes.com": 0.87, "washingtonpost.com": 0.87,
        "who.int": 0.95, "cdc.gov": 0.95, "nih.gov": 0.94, "nature.com": 0.94,
    }
    UNRELIABLE = {
        "infowars.com": 0.12, "beforeitsnews.com": 0.15, "naturalnews.com": 0.18
    }
    if not domain:
        return 0.45
    for d, s in TRUSTED.items():
        if d in domain:
            return s
    for d, s in UNRELIABLE.items():
        if d in domain:
            return s
    if domain.endswith(".gov"):
        return 0.85
    if domain.endswith(".edu"):
        return 0.80
    if domain.endswith(".org"):
        return 0.68
    return 0.5


# -------------------------
# Claim extraction (light)
# -------------------------
def extract_claim(text: str) -> str:
    """
    If text is short, use the whole text as claim.
    If long, pick the most assertive sentence (heuristic).
    """
    if not text:
        return ""
    text = clean_text(text)
    tokens = text.split()
    if len(tokens) <= 20:
        return text  # user likely pasted a headline/claim
    # split into sentences heuristically
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # prefer sentences with numbers or named entities (heuristic)
    scored = []
    for s in sentences:
        score = 0
        if re.search(r'\d', s):
            score += 1
        if '"' in s or "'" in s:
            score += 1
        score += min(len(s.split()) / 20.0, 1.0)
        scored.append((score, s))
    scored.sort(reverse=True)
    return scored[0][1].strip() if scored else text


# -------------------------
# Google Custom Search
# -------------------------
def google_search(query: str, num: int = MAX_GOOGLE_RESULTS) -> List[Dict[str, Any]]:
    """Return list of items: {title, link, snippet}."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("Google API key or CSE ID not configured in environment.")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(10, num)
    }
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", []) or []
    results = []
    for it in items:
        results.append({
            "title": it.get("title"),
            "url": it.get("link"),
            "snippet": it.get("snippet")
        })
    return results


# -------------------------
# Web scraping + text extraction
# -------------------------
def scrape_page(url: str, max_chars: int = MAX_SCRAPED_CHARS) -> str:
    """Fetch page and return cleaned text. Use basic requests + BeautifulSoup."""
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        # network or SSL error
        raise RuntimeError(f"Failed to fetch {url}: {e}")
    if r.status_code != 200 or not r.text:
        raise RuntimeError(f"Non-200 or empty response for {url}: status={r.status_code}")
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    # Prefer article-like content: <article> then <p>
    content = ""
    article = soup.find("article")
    if article:
        content = clean_text(article.get_text(separator=" ", strip=True))
    if not content:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        content = " ".join([p for p in paragraphs if p])
    content = clean_text(content)
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    return content


# -------------------------
# Light reasoning / NLI heuristic
# -------------------------
NEGATION_WORDS = {"not", "no", "never", "none", "n't", "cannot", "without"}
REFUTE_HINTS = {"false", "fake", "debunked", "disproved", "no evidence", "misleading"}
SUPPORT_HINTS = {"confirmed", "found", "reported", "shows", "study", "studies", "evidence"}

def sentence_tokens(s: str) -> set:
    return set(re.sub(r'[^0-9a-zA-Z]+', ' ', s).lower().split())

def score_claim_vs_evidence(claim: str, evidence_text: str) -> Dict[str, Any]:
    """
    Returns a dict: {label: 'supports'|'refutes'|'neutral', score: float}
    Heuristic:
      - token overlap proportion between claim tokens and evidence tokens (claim-centric)
      - presence of obvious refute/support cue words adjusts label/score
      - negation handling: if evidence contains negation + claim keywords -> refute
    """
    claim = clean_text(claim)
    evidence = clean_text(evidence_text)
    if not evidence:
        return {"label": "neutral", "score": 0.0}
    c_tokens = sentence_tokens(claim)
    e_tokens = sentence_tokens(evidence)
    if not c_tokens:
        return {"label": "neutral", "score": 0.0}
    overlap = len(c_tokens & e_tokens) / max(1, len(c_tokens))
    # base score (0..1)
    base = overlap
    # cue words influence
    evidence_lower = evidence.lower()
    support_hits = sum(1 for w in SUPPORT_HINTS if w in evidence_lower)
    refute_hits = sum(1 for w in REFUTE_HINTS if w in evidence_lower)
    # negation detection
    negation = any(w in evidence_lower for w in NEGATION_WORDS)
    # heuristics
    score = base
    if support_hits > 0:
        score = max(score, min(0.6 + 0.1*support_hits, 0.95))
        label = "supports"
    elif refute_hits > 0 or (negation and overlap > 0.15):
        # evidence contains refutation cue or negation present with some overlap
        score = max(score, min(0.6 + 0.1*refute_hits, 0.95))
        label = "refutes"
    else:
        label = "supports" if score > 0.35 else "neutral"
    # clamp score
    score = max(0.0, min(1.0, float(score)))
    return {"label": label, "score": score}


# -------------------------
# Orchestration: single detect function
# -------------------------
def analyze_claim_pipeline(input_text: str, input_url: Optional[str], logs: List[str]) -> Dict[str, Any]:
    start = time.time()
    # 1) decide claim text
    text_for_claim = input_text or ""
    if input_url and not text_for_claim:
        logs.append(f"Attempting to extract text from provided URL before claim extraction: {input_url}")
        try:
            scraped = scrape_page(input_url, max_chars=1500)
            text_for_claim = scraped
            logs.append("Extracted text from URL for claim extraction.")
        except Exception as e:
            logs.append(f"Could not extract from URL: {e}. Proceeding with original text (if any).")
    claim = extract_claim(text_for_claim)
    claim = clean_text(claim)
    if not claim:
        raise RuntimeError("No claim text available after extraction.")
    logs.append(f"Claim chosen for verification: '{claim[:200]}'")

    # 2) google search
    logs.append("Running Google Custom Search for evidence (only Google).")
    try:
        google_results = google_search(claim, num=MAX_GOOGLE_RESULTS)
        logs.append(f"Google returned {len(google_results)} results.")
    except Exception as e:
        logs.append(f"Google search failed: {e}")
        raise RuntimeError(f"Google search failed: {e}")

    if not google_results:
        logs.append("No search results returned by Google.")
        # we'll still return Cannot Verify but include logs
        return {
            "claim": claim,
            "verdict": "Cannot Verify",
            "confidence": 0.0,
            "explanation": "Google returned no search results for the claim.",
            "sources": [],
            "processing_time": time.time() - start,
            "logs": logs
        }

    # 3) scrape top results and score each
    source_objs: List[Dict[str, Any]] = []
    for idx, res in enumerate(google_results[:MAX_GOOGLE_RESULTS]):
        url = res.get("url")
        title = res.get("title") or ""
        snippet = res.get("snippet") or ""
        logs.append(f"Scraping result {idx+1}: {url}")
        scraped_text = ""
        try:
            scraped_text = scrape_page(url, max_chars=MAX_SCRAPED_CHARS)
            logs.append(f"Scraped {len(scraped_text)} chars from {url}")
        except Exception as e:
            logs.append(f"Scrape failed for {url}: {e}")
            # fall back to snippet if scrape fails
            scraped_text = snippet or ""

        domain = domain_from_url(url)
        dscore = score_domain(domain)
        nli = score_claim_vs_evidence(claim, scraped_text or snippet or "")
        source_objs.append({
            "title": clean_text(title),
            "url": url,
            "snippet": clean_text(snippet),
            "scraped_text": scraped_text,
            "domain_score": float(dscore),
            "source_label": nli["label"],
            "source_score": float(nli["score"])
        })

    # 4) aggregate evidence weighted by domain_score
    support_weight = 0.0
    refute_weight = 0.0
    total_weight = 0.0
    for s in source_objs:
        weight = s["domain_score"]  # domain reliability as weight
        total_weight += weight
        if s["source_label"] == "supports":
            support_weight += s["source_score"] * weight
        elif s["source_label"] == "refutes":
            refute_weight += s["source_score"] * weight
        # neutral does not add

    if total_weight == 0:
        verdict = "Cannot Verify"
        confidence = 0.0
        logs.append("Total aggregated weight is zero; cannot compute verdict.")
    else:
        support_norm = support_weight / total_weight
        refute_norm = refute_weight / total_weight
        logs.append(f"Aggregate support score {support_norm:.3f}, refute score {refute_norm:.3f}")

        # Decision thresholds (tunable)
        if support_norm > refute_norm and support_norm >= 0.55:
            verdict = "Supported"
            confidence = support_norm
        elif refute_norm > support_norm and refute_norm >= 0.55:
            verdict = "Refuted"
            confidence = refute_norm
        elif max(support_norm, refute_norm) >= 0.45:
            # ambiguous but leaning
            verdict = "Needs More Evidence"
            confidence = max(support_norm, refute_norm)
        else:
            verdict = "Needs More Evidence"
            confidence = max(support_norm, refute_norm)

    # 5) short explanation (1 paragraph)
    if verdict == "Supported":
        explanation = (
            f"The claim appears to be supported by multiple sources (weighted score {confidence:.2f}). "
            "High-reliability domains referenced similar statements; see sources for details."
        )
    elif verdict == "Refuted":
        explanation = (
            f"The claim appears contradicted by available sources (weighted score {confidence:.2f}). "
            "Evidence from scraped pages suggests the claim is false or misleading; see sources."
        )
    elif verdict == "Needs More Evidence":
        explanation = (
            "Available search results provide limited or mixed evidence. "
            "Some sources partially align with the claim but do not conclusively support it. "
            "Review the scraped sources below for details."
        )
    else:
        explanation = "Could not verify the claim due to lack of returned evidence."

    processing_time = time.time() - start

    # wrapper result
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": float(round(confidence, 3)),
        "explanation": explanation,
        "sources": source_objs,
        "processing_time": processing_time,
        "logs": logs
    }


# -------------------------
# FastAPI endpoints
# -------------------------
@app.get("/health")
def health():
    ok = bool(GOOGLE_API_KEY and GOOGLE_CSE_ID)
    return {
        "status": "healthy" if ok else "missing-google-keys",
        "google_configured": ok,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    logs: List[str] = []
    start_time = time.time()

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logs.append("Google API key or CSE ID missing in environment.")
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing Google API key / CSE ID.")

    if not req.text and not req.url:
        raise HTTPException(status_code=400, detail="Please provide 'text' or 'url' in the request.")

    try:
        # unify input text: prefer provided text; if only url given, we'll try to extract
        text_input = req.text or ""
        url_input = req.url or ""

        result = analyze_claim_pipeline(text_input, url_input, logs)
        # map sources to SourceOut model
        sources_out: List[SourceOut] = []
        for s in result.get("sources", []):
            sources_out.append(SourceOut(
                title=s.get("title"),
                url=s.get("url"),
                snippet=s.get("snippet"),
                scraped_text=s.get("scraped_text"),
                domain_score=s.get("domain_score", 0.0),
                source_label=s.get("source_label", "neutral"),
                source_score=s.get("source_score", 0.0)
            ))

        resp = DetectResponse(
            claim=result.get("claim", ""),
            verdict=result.get("verdict", "Cannot Verify"),
            confidence=float(result.get("confidence", 0.0)),
            explanation=result.get("explanation", ""),
            sources=sources_out,
            processing_time=float(result.get("processing_time", 0.0)),
            timestamp=datetime.utcnow().isoformat(),
            logs=result.get("logs", []) + logs
        )
        return resp

    except HTTPException:
        raise
    except Exception as e:
        # log and show concise error
        logs.append(f"Unexpected server error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

