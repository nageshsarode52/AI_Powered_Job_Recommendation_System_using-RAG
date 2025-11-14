# src/retriever.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global caching variables
_embed_model = None
_index = None
_meta = None

def load_embedding_model():
    """Load the embedding model once."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMB_MODEL)
    return _embed_model

def load_store(index_path="data/faiss.index", meta_path="data/faiss_meta.pkl"):
    """Load FAISS index and metadata into memory."""
    global _index, _meta
    if _index is None or _meta is None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found. Check your data folder.")
        _index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            _meta = pickle.load(f)
    return _index, _meta

def embed_texts(texts):
    """Embed text list using sentence-transformers."""
    model = load_embedding_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs

def retrieve_top_k(query, k=5, index_path="data/faiss.index", meta_path="data/faiss_meta.pkl"):
    """Retrieve top-k matching jobs for a given profile."""
    idx, meta = load_store(index_path, meta_path)
    q_emb = embed_texts([query]).astype('float32')
    # simple L2 search
    D, I = idx.search(q_emb, k)
    results = []
    for i, dist in zip(I[0], D[0]):
        m = meta[i]
        results.append({
            "title": m.get("title"),
            "company": m.get("company"),
            "location": m.get("location"),
            "url": m.get("url"),
            "description": m.get("description"),
            "score": float(dist)
        })
    return results

def compute_match_scores(retrieved):
    """Convert L2 distances into 0–100 match scores (higher = better)."""
    if not retrieved:
        return []
    dists = [r['score'] for r in retrieved]
    mind, maxd = min(dists), max(dists)
    out = []
    for r in retrieved:
        if maxd - mind == 0:
            score = 70
        else:
            score = int(100 * (1 - (r['score'] - mind) / (maxd - mind)))
        r2 = r.copy()
        r2['match_score'] = score
        out.append(r2)
    return out


def explain_recommendations(profile, jobs):
    """
    Robust Gemini caller with automatic-model fallback and safe error handling.
    Returns a numbered list (string) or a friendly info message when Gemini isn't available.
    """
    # Lazy import so missing packages don't break module import
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.schema import HumanMessage
    except Exception:
        return "Gemini reasoning not available: required packages not installed."

    # Candidate models to try (ordered) - using the model names available in your account
    candidates = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "models/chat-bison@001",
    ]

    llm = None
    last_err = None
    for cand in candidates:
        try:
            llm = ChatGoogleGenerativeAI(model=cand)
            # if construction succeeded, break and use this model
            break
        except Exception as e:
            last_err = e
            llm = None

    if llm is None:
        return f"Gemini reasoning not available: no compatible model found. Last error: {last_err}"

    # Build prompt (truncate descriptions to avoid token bloat)
    job_text = ""
    for i, j in enumerate(jobs, start=1):
        desc = (j.get("description") or "")[:700].replace("\n", " ")
        job_text += f"{i}. {j.get('title','N/A')} at {j.get('company','N/A')} ({j.get('location','N/A')})\nDescription: {desc}\n\n"

    prompt = f"""
You are an AI career advisor.

User profile:
{profile}

Job postings:
{job_text}

For each job, explain in ONE short sentence why it fits the user's skills and interests.
Return the answers as a numbered list (1, 2, 3, ...).
"""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        # resp.content is the generated string
        return resp.content
    except Exception as e:
        return f"Gemini reasoning failed at runtime (LLM call error): {e}"