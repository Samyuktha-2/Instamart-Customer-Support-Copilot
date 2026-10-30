import glob
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm

POLICY_DIR = os.path.join(ROOT, "data", "policies")

POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT for a grocery delivery company. "
    "Answer only from the supplied context. "
    "If the context does not contain the answer, say you could not find the information."
)


def load_docs():
    docs = {}
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs


def chunk_text(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):      # this chunk reaches the last word
            break
        start += step
    return chunks


def chunk_docs(size=40, overlap=10):
    out = []
    for name, text in load_docs().items():
        for c in chunk_text(text, size, overlap):
            out.append({"source": name, "text": c})
    return out


class Index:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_docs(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])


_cache = {}


def get_index(size=40, overlap=10):
    key = (size, overlap)
    if key not in _cache:
        _cache[key] = Index(size, overlap)
    return _cache[key]


def retrieve(query, k=3, index=None):
    idx = index or get_index()
    sims = cosine_similarity(idx.vec.transform([query]), idx.matrix)[0]
    order = sims.argsort()[::-1][:k]
    return [{"source": idx.chunks[i]["source"], "score": float(sims[i]),
             "text": idx.chunks[i]["text"]} for i in order]


def answer(question, index=None):
    hits = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
    ]
    resp = mock_llm.chat(messages)
    return {"answer": resp.choices[0].message.content,
            "sources": [h["source"] for h in hits]}