from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

app = FastAPI()

# Load NLI Model
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

class Input(BaseModel):
    text: str = None
    url: str = None



def extract_text_from_url(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        full_text = " ".join([p.text for p in paragraphs])
        return full_text
    except:
        return None



def extract_main_claim(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    for s in sentences:
        words = s.strip().split()
        if 6 < len(words) < 40:
            return s.strip()
    return " ".join(text.split()[:20])



def google_search(query):
    results = []
    GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("⚠ Missing API keys in .env")
        return []

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query
    }

    print("\n🔍 GOOGLE QUERY:", params)  # Debug Log

    try:
        response = requests.get(GOOGLE_URL, params=params).json()
        print("🔎 GOOGLE RAW RESPONSE:", response)  # Debug Log

        if "items" in response:
            for item in response["items"]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", "")
                })

    except Exception as e:
        print("❌ Google Error:", e)

    return results



def classify_claim(claim, evidence):
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    labels = ["contradiction", "neutral", "entailment"]

    return labels[int(torch.argmax(probs))], float(torch.max(probs))



@app.post("/detect")
def detect(input: Input):

    # Step 1: Extract text
    text = extract_text_from_url(input.url) if input.url else input.text

    if not text:
        return {"error": "Unable to extract or process text."}

    # Step 2: Extract main claim
    claim = extract_main_claim(text)
    print("\n🟦 EXTRACTED CLAIM:", claim)

    # Step 3: Retrieve facts
    evidence_list = google_search(claim)

    if not evidence_list:
        return {
            "category": "Needs Verification",
            "confidence": 0.0,
            "claim": claim,
            "matched_evidence": None,
            "verified_sources": [],
            "reason": "No credible evidence found."
        }

   
    top_evidence = evidence_list[0]

    
    verdict, confidence = classify_claim(claim, top_evidence["snippet"])

    verdict_map = {
        "entailment": "Authentic Information",
        "contradiction": "Likely Misinformation",
        "neutral": "Needs More Evidence"
    }

    return {
        "category": verdict_map[verdict],
        "confidence": confidence,
        "claim": claim,
        "matched_evidence": top_evidence,
        "verified_sources": evidence_list[:5]
    }
