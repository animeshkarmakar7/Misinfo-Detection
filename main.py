# ============================================================================
# CrisisGuard: Agentic AI Misinformation Detection (FREE API Version)
# Optimized for Hackathon with Zero-Cost Infrastructure
# ============================================================================

"""
SETUP:
pip install langgraph langchain langchain-community fastapi streamlit requests 
pip install beautifulsoup4 python-dotenv sentence-transformers faiss-cpu
pip install newspaper3k lxml_html_clean transformers torch spacy

python -m spacy download en_core_web_sm

.env file:
GOOGLE_API_KEY=your_key
GOOGLE_CSE_ID=your_cse_id
HUGGINGFACE_API_KEY=optional_for_rate_limits
"""

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
import requests
from datetime import datetime
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# SHARED STATE
# ============================================================================

class AgentState(TypedDict):
    """Shared state across all agents"""
    input_text: str
    input_url: str
    
    # Agent outputs
    extracted_claims: List[Dict]
    evidence_results: List[Dict]
    scraped_content: List[Dict]
    credibility_scores: Dict
    reasoning_analysis: Dict
    
    # Final output
    verdict: str
    confidence: float
    explanation: str
    sources: List[Dict]
    agent_logs: List[str]
    
    # Tracking
    google_api_calls: int
    processing_time: float


# ============================================================================
# AGENT 1: CLAIM EXTRACTION (Using Free NLP)
# ============================================================================

class ClaimExtractionAgent:
    """Extract claims using rule-based + spaCy (FREE)"""
    
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("⚠️ spaCy not available, using rule-based extraction")
    
    def extract_claims_spacy(self, text: str) -> List[Dict]:
        """Use spaCy for entity-aware extraction"""
        if not self.nlp:
            return self.extract_claims_rules(text)
        
        doc = self.nlp(text)
        claims = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Filter out questions, very short sentences
            if len(sent_text.split()) < 5 or sent_text.endswith("?"):
                continue
            
            # Extract entities
            entities = [ent.text for ent in sent.ents]
            
            # Identify claim type
            claim_type = "general"
            if any(ent.label_ in ["PERCENT", "MONEY", "QUANTITY"] for ent in sent.ents):
                claim_type = "statistical"
            elif any(ent.label_ == "DATE" for ent in sent.ents):
                claim_type = "temporal"
            elif any(ent.label_ in ["PERSON", "ORG"] for ent in sent.ents):
                claim_type = "attribution"
            
            # Priority based on verifiability
            priority = "high" if entities and claim_type != "general" else "medium"
            
            claims.append({
                "claim": sent_text,
                "entities": entities,
                "type": claim_type,
                "priority": priority
            })
        
        return claims
    
    def extract_claims_rules(self, text: str) -> List[Dict]:
        """Fallback: Rule-based extraction"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 5:
                continue
            
            # Simple heuristics
            has_numbers = bool(re.search(r'\d+', sent))
            has_quotes = '"' in sent or "'" in sent
            
            priority = "high" if (has_numbers or has_quotes) else "medium"
            
            claims.append({
                "claim": sent,
                "entities": [],
                "type": "general",
                "priority": priority
            })
        
        return claims[:5]  # Limit to top 5
    
    def __call__(self, state: AgentState) -> AgentState:
        state["agent_logs"].append("🔍 [Extraction] Analyzing text structure...")
        
        text = state.get("input_text", "")
        if len(text) < 10:
            state["extracted_claims"] = []
            state["agent_logs"].append("⚠️ [Extraction] Text too short")
            return state
        
        claims = self.extract_claims_spacy(text)
        
        # Sort by priority
        claims.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])
        
        state["extracted_claims"] = claims[:3]  # Top 3 claims
        state["agent_logs"].append(f"✅ [Extraction] Found {len(claims)} claims")
        
        return state


# ============================================================================
# AGENT 2: EVIDENCE RETRIEVAL (Multi-source with caching)
# ============================================================================

class EvidenceRetrievalAgent:
    """Search multiple sources with intelligent caching"""
    
    def __init__(self, google_api_key: str, google_cse_id: str):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.cache = {}  # Simple in-memory cache
    
    def search_google(self, query: str, num_results: int = 5) -> List[Dict]:
        """Google Custom Search (100 free queries/day)"""
        
        # Check cache first
        cache_key = f"google:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": num_results
        }
        
        try:
            response = requests.get(url, params=params, timeout=10).json()
            
            if "items" not in response:
                return []
            
            results = []
            for item in response["items"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google"
                })
            
            # Cache results
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            print(f"❌ Google Search Error: {e}")
            return []
    
    def search_duckduckgo(self, query: str) -> List[Dict]:
        """DuckDuckGo Instant Answer API (FREE, unlimited)"""
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10).json()
            
            results = []
            
            # Abstract (Wikipedia-style summary)
            if response.get("Abstract"):
                results.append({
                    "title": response.get("Heading", "Summary"),
                    "url": response.get("AbstractURL", ""),
                    "snippet": response.get("Abstract", ""),
                    "source": "duckduckgo"
                })
            
            # Related topics
            for topic in response.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:50],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "source": "duckduckgo"
                    })
            
            return results
            
        except Exception as e:
            print(f"⚠️ DuckDuckGo Error: {e}")
            return []
    
    def build_query(self, claim: Dict) -> str:
        """Build optimized search query"""
        claim_text = claim["claim"]
        entities = claim.get("entities", [])
        
        # Include top entities for better results
        if entities:
            query = f"{' '.join(entities[:2])} {claim_text}"
        else:
            query = claim_text
        
        # Limit query length
        return " ".join(query.split()[:15])
    
    def __call__(self, state: AgentState) -> AgentState:
        state["agent_logs"].append("🔎 [Evidence] Searching multiple sources...")
        
        claims = state.get("extracted_claims", [])
        all_evidence = []
        google_calls = 0
        
        for i, claim in enumerate(claims[:2]):  # Limit to 2 claims to save API quota
            query = self.build_query(claim)
            state["agent_logs"].append(f"🔎 [Evidence] Query {i+1}: '{query[:50]}...'")
            
            # Try DuckDuckGo first (FREE)
            ddg_results = self.search_duckduckgo(query)
            all_evidence.extend(ddg_results)
            
            # Use Google only if DuckDuckGo fails or for high-priority claims
            if len(ddg_results) < 2 and claim["priority"] == "high":
                google_results = self.search_google(query, num_results=3)
                all_evidence.extend(google_results)
                google_calls += 1
        
        state["evidence_results"] = all_evidence
        state["google_api_calls"] = google_calls
        state["agent_logs"].append(
            f"✅ [Evidence] Retrieved {len(all_evidence)} sources "
            f"(Google calls: {google_calls})"
        )
        
        return state


# ============================================================================
# AGENT 3: WEB SCRAPING (Extract full context)
# ============================================================================

class WebScrapingAgent:
    """Scrape evidence pages for deeper analysis"""
    
    def scrape_url(self, url: str) -> Optional[Dict]:
        """Extract text from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Get paragraphs for better quality
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
            
            return {
                "url": url,
                "full_text": text[:2000],  # Limit size
                "paragraphs": paragraphs[:5],
                "word_count": len(text.split())
            }
            
        except Exception as e:
            print(f"⚠️ Scraping failed for {url}: {e}")
            return None
    
    def __call__(self, state: AgentState) -> AgentState:
        state["agent_logs"].append("📄 [Scraping] Extracting full content...")
        
        evidence = state.get("evidence_results", [])
        scraped = []
        
        # Scrape top 3 credible sources
        for item in evidence[:3]:
            url = item.get("url", "")
            if url:
                content = self.scrape_url(url)
                if content:
                    scraped.append(content)
        
        state["scraped_content"] = scraped
        state["agent_logs"].append(f"✅ [Scraping] Extracted {len(scraped)} pages")
        
        return state


# ============================================================================
# AGENT 4: CREDIBILITY SCORING (Enhanced with multiple signals)
# ============================================================================

class CredibilityAgent:
    """Multi-factor source credibility scoring"""
    
    # Expanded trusted domains with scores
    TRUSTED_DOMAINS = {
        # News agencies
        "reuters.com": 0.95, "apnews.com": 0.95, "bbc.com": 0.92,
        "theguardian.com": 0.88, "nytimes.com": 0.87, "washingtonpost.com": 0.87,
        "cnn.com": 0.82, "bloomberg.com": 0.85, "economist.com": 0.90,
        
        # Scientific
        "nature.com": 0.98, "science.org": 0.98, "sciencedirect.com": 0.92,
        "pubmed.ncbi.nlm.nih.gov": 0.95, "arxiv.org": 0.85,
        
        # Government/International
        "who.int": 0.95, "cdc.gov": 0.95, "fda.gov": 0.93,
        "un.org": 0.92, "europa.eu": 0.90,
        
        # Fact-checkers
        "snopes.com": 0.90, "factcheck.org": 0.92, "politifact.com": 0.88,
        "fullfact.org": 0.88
    }
    
    # Red flag domains
    UNRELIABLE_DOMAINS = {
        "beforeitsnews.com": 0.15, "infowars.com": 0.10,
        "naturalnews.com": 0.15, "worldtruth.tv": 0.10
    }
    
    def score_domain(self, url: str) -> float:
        """Domain reputation scoring"""
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")
        except:
            return 0.40
        
        # Check trusted domains
        if domain in self.TRUSTED_DOMAINS:
            return self.TRUSTED_DOMAINS[domain]
        
        # Check unreliable domains
        if domain in self.UNRELIABLE_DOMAINS:
            return self.UNRELIABLE_DOMAINS[domain]
        
        # Partial matching for subdomains
        for trusted, score in self.TRUSTED_DOMAINS.items():
            if trusted in domain:
                return score * 0.95  # Slight penalty for subdomain
        
        # TLD-based scoring
        if domain.endswith('.gov'):
            return 0.85
        elif domain.endswith('.edu'):
            return 0.80
        elif domain.endswith('.org'):
            return 0.70
        
        return 0.50  # Unknown domain
    
    def score_content_quality(self, scraped: Dict) -> float:
        """Content quality signals"""
        score = 0.5
        
        word_count = scraped.get("word_count", 0)
        paragraphs = scraped.get("paragraphs", [])
        
        # Length signal (not too short, not too long)
        if 300 < word_count < 2000:
            score += 0.15
        elif word_count >= 200:
            score += 0.05
        
        # Structure signal
        if len(paragraphs) >= 3:
            score += 0.10
        
        return min(1.0, score)
    
    def __call__(self, state: AgentState) -> AgentState:
        state["agent_logs"].append("⚖️ [Credibility] Scoring sources...")
        
        evidence = state.get("evidence_results", [])
        scraped = state.get("scraped_content", [])
        
        credibility_scores = {}
        
        for item in evidence:
            url = item.get("url", "")
            
            # Domain score
            domain_score = self.score_domain(url)
            
            # Content quality score (if scraped)
            content_score = 0.5
            for s in scraped:
                if s.get("url") == url:
                    content_score = self.score_content_quality(s)
                    break
            
            # Combined score (weighted)
            final_score = (domain_score * 0.7) + (content_score * 0.3)
            
            credibility_scores[url] = {
                "score": final_score,
                "domain_score": domain_score,
                "content_score": content_score,
                "category": "High" if final_score > 0.75 else 
                           "Medium" if final_score > 0.55 else "Low"
            }
        
        state["credibility_scores"] = credibility_scores
        
        high_cred = sum(1 for s in credibility_scores.values() if s["score"] > 0.75)
        state["agent_logs"].append(
            f"⚖️ [Credibility] {high_cred}/{len(credibility_scores)} high-credibility sources"
        )
        
        return state


# ============================================================================
# AGENT 5: NLI REASONING (Using Free Hugging Face Models)
# ============================================================================

class ReasoningAgent:
    """Natural Language Inference using free transformers"""
    
    def __init__(self):
        try:
            from transformers import pipeline
            
            # Use free NLI model (runs locally)
            self.nli_model = pipeline(
                "text-classification",
                model="microsoft/deberta-v3-base-tasksource-nli",
                device=-1  # CPU
            )
            self.available = True
        except Exception as e:
            print(f"⚠️ NLI model not available: {e}")
            self.available = False
    
    def analyze_claim_evidence(self, claim: str, evidence: str) -> Dict:
        """NLI classification"""
        if not self.available:
            return {"label": "neutral", "score": 0.5}
        
        try:
            # Truncate for model limits
            claim = claim[:256]
            evidence = evidence[:256]
            
            result = self.nli_model(f"{claim} [SEP] {evidence}")[0]
            
            # Map labels
            label_map = {
                "entailment": "supports",
                "contradiction": "refutes",
                "neutral": "neutral"
            }
            
            return {
                "label": label_map.get(result["label"].lower(), "neutral"),
                "score": result["score"]
            }
        except:
            return {"label": "neutral", "score": 0.5}
    
    def aggregate_evidence(self, claim: str, evidence_list: List[Dict], 
                          credibility_scores: Dict) -> Dict:
        """Aggregate NLI results weighted by credibility"""
        
        support_score = 0.0
        refute_score = 0.0
        total_weight = 0.0
        
        for evidence in evidence_list[:5]:
            snippet = evidence.get("snippet", "")
            url = evidence.get("url", "")
            
            if not snippet:
                continue
            
            # Get NLI result
            nli_result = self.analyze_claim_evidence(claim, snippet)
            
            # Get credibility weight
            cred = credibility_scores.get(url, {}).get("score", 0.5)
            
            # Weighted scoring
            if nli_result["label"] == "supports":
                support_score += nli_result["score"] * cred
            elif nli_result["label"] == "refutes":
                refute_score += nli_result["score"] * cred
            
            total_weight += cred
        
        if total_weight == 0:
            return {"verdict": "neutral", "confidence": 0.0}
        
        # Normalize
        support_score /= total_weight
        refute_score /= total_weight
        
        # Determine verdict
        if support_score > refute_score and support_score > 0.6:
            return {"verdict": "supported", "confidence": support_score}
        elif refute_score > support_score and refute_score > 0.6:
            return {"verdict": "refuted", "confidence": refute_score}
        else:
            return {"verdict": "neutral", "confidence": 0.5}
    
    def __call__(self, state: AgentState) -> AgentState:
        state["agent_logs"].append("🧠 [Reasoning] Analyzing claim-evidence alignment...")
        
        claims = state.get("extracted_claims", [])
        evidence = state.get("evidence_results", [])
        credibility = state.get("credibility_scores", {})
        
        if not claims or not evidence:
            state["reasoning_analysis"] = {"verdict": "insufficient", "confidence": 0.0}
            return state
        
        # Analyze primary claim
        main_claim = claims[0]["claim"]
        result = self.aggregate_evidence(main_claim, evidence, credibility)
        
        state["reasoning_analysis"] = result
        state["agent_logs"].append(
            f"🧠 [Reasoning] Verdict: {result['verdict']} "
            f"(confidence: {result['confidence']:.2f})"
        )
        
        return state


# ============================================================================
# AGENT 6: SYNTHESIS & EXPLANATION
# ============================================================================

class SynthesisAgent:
    """Generate final verdict with explanation"""
    
    def generate_explanation(self, state: AgentState) -> str:
        """Build human-readable explanation"""
        reasoning = state.get("reasoning_analysis", {})
        evidence = state.get("evidence_results", [])
        credibility = state.get("credibility_scores", {})
        
        verdict = reasoning.get("verdict", "neutral")
        
        # Count high-credibility sources
        high_cred_sources = [
            e for e in evidence 
            if credibility.get(e.get("url", ""), {}).get("score", 0) > 0.75
        ]
        
        explanations = {
            "supported": (
                f"The claim is supported by {len(high_cred_sources)} high-credibility sources. "
                f"Evidence from trusted outlets aligns with the claim's assertions."
            ),
            "refuted": (
                f"The claim contradicts information from {len(high_cred_sources)} credible sources. "
                f"Available evidence does not support the claim's assertions."
            ),
            "neutral": (
                f"Insufficient evidence to verify. Found {len(evidence)} sources, but "
                f"evidence is ambiguous or lacks high-credibility confirmation."
            ),
            "insufficient": (
                "Unable to locate sufficient credible sources to verify this claim. "
                "More investigation is needed."
            )
        }
        
        return explanations.get(verdict, "Analysis inconclusive.")
    
    def __call__(self, state: AgentState) -> AgentState:
        state["agent_logs"].append("📊 [Synthesis] Generating final verdict...")
        
        reasoning = state.get("reasoning_analysis", {})
        evidence = state.get("evidence_results", [])
        
        verdict_map = {
            "supported": "Authentic Information",
            "refuted": "Likely Misinformation",
            "neutral": "Needs More Evidence",
            "insufficient": "Cannot Verify"
        }
        
        reasoning_verdict = reasoning.get("verdict", "insufficient")
        confidence = reasoning.get("confidence", 0.0)
        
        state["verdict"] = verdict_map[reasoning_verdict]
        state["confidence"] = confidence
        state["explanation"] = self.generate_explanation(state)
        state["sources"] = evidence[:5]
        
        state["agent_logs"].append(
            f"✅ [Synthesis] Final: {state['verdict']} ({confidence*100:.0f}%)"
        )
        
        return state


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

class CrisisGuardWorkflow:
    """Main agentic workflow"""
    
    def __init__(self, google_api_key: str, google_cse_id: str):
        self.extraction = ClaimExtractionAgent()
        self.evidence = EvidenceRetrievalAgent(google_api_key, google_cse_id)
        self.scraping = WebScrapingAgent()
        self.credibility = CredibilityAgent()
        self.reasoning = ReasoningAgent()
        self.synthesis = SynthesisAgent()
        
        self.workflow = self._build_graph()
    
    def _build_graph(self):
        graph = StateGraph(AgentState)
        
        # Add agents
        graph.add_node("extraction", self.extraction)
        graph.add_node("evidence", self.evidence)
        graph.add_node("scraping", self.scraping)
        graph.add_node("credibility", self.credibility)
        graph.add_node("reasoning", self.reasoning)
        graph.add_node("synthesis", self.synthesis)
        
        # Sequential flow
        graph.add_edge("extraction", "evidence")
        graph.add_edge("evidence", "scraping")
        graph.add_edge("scraping", "credibility")
        graph.add_edge("credibility", "reasoning")
        graph.add_edge("reasoning", "synthesis")
        graph.add_edge("synthesis", END)
        
        graph.set_entry_point("extraction")
        
        return graph.compile()
    
    def run(self, input_text: str = None, input_url: str = None) -> Dict:
        """Execute workflow"""
        start_time = datetime.now()
        
        initial_state = {
            "input_text": input_text or "",
            "input_url": input_url or "",
            "agent_logs": [],
            "extracted_claims": [],
            "evidence_results": [],
            "scraped_content": [],
            "credibility_scores": {},
            "reasoning_analysis": {},
            "verdict": "",
            "confidence": 0.0,
            "explanation": "",
            "sources": [],
            "google_api_calls": 0,
            "processing_time": 0.0
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        final_state["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        return final_state


# ============================================================================
# FASTAPI BACKEND
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CrisisGuard Agentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize workflow
workflow = CrisisGuardWorkflow(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

class DetectionRequest(BaseModel):
    text: str = None
    url: str = None

@app.post("/detect")
async def detect_misinformation(request: DetectionRequest):
    try:
        # Handle URL extraction
        if request.url:
            from bs4 import BeautifulSoup
            try:
                response = requests.get(request.url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = [p.get_text() for p in soup.find_all('p')]
                request.text = ' '.join(paragraphs[:10])  # First 10 paragraphs
            except:
                raise HTTPException(400, "Failed to extract text from URL")
        
        if not request.text:
            raise HTTPException(400, "No text provided")
        
        # Run workflow
        result = workflow.run(input_text=request.text)
        
        return {
            "category": result["verdict"],
            "confidence": result["confidence"],
            "claim": result["extracted_claims"][0]["claim"] if result["extracted_claims"] else request.text[:200],
            "matched_evidence": result["sources"][0] if result["sources"] else None,
            "verified_sources": result["sources"],
            "explanation": result["explanation"],
            "agent_logs": result["agent_logs"],
            "processing_time": result["processing_time"],
            "google_api_calls": result["google_api_calls"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": 6}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)