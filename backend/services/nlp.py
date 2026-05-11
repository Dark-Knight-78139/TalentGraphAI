import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load models safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to load sentence transformer: {e}")
    model = None

def extract_entities(text: str) -> list[str]:
    """
    Extracts potential skill-like entities from text using spaCy.
    """
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "GPE"]:
            skills.add(ent.text)
    
    # Also grab capitalized nouns as potential skills
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"] and token.is_title:
            skills.add(token.text)
            
    return list(skills)

def compute_similarity(skill1: str, skill2: str) -> float:
    """
    Computes cosine similarity between two skills.
    """
    if model is None:
        return 0.0
    
    embeddings1 = model.encode([skill1])
    embeddings2 = model.encode([skill2])
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores[0][0].item()

def match_skills(required: list[str], candidate: list[str], threshold: float = 0.6) -> dict:
    """
    Matches required skills to candidate skills using semantic similarity.
    Returns matched and missing skills.
    """
    matched = {}
    missing = []
    
    if model is None:
        return {"matched": {r: [c for c in candidate if c.lower() == r.lower()] for r in required}, "missing": required}
        
    req_embeddings = model.encode(required)
    cand_embeddings = model.encode(candidate)
    
    cosine_scores = util.cos_sim(req_embeddings, cand_embeddings)
    
    for i, req in enumerate(required):
        best_match_idx = np.argmax(cosine_scores[i]).item()
        best_score = cosine_scores[i][best_match_idx].item()
        
        if best_score >= threshold:
            matched[req] = candidate[best_match_idx]
        else:
            missing.append(req)
            
    return {"matched": matched, "missing": missing}
