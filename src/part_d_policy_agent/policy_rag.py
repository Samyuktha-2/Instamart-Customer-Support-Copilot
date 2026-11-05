
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = PROJECT_ROOT / "data" / "policies"

CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10


def chunk_documents(overlap=DEFAULT_OVERLAP):
    """
    Split all policy documents into 40-word chunks.

    Each subsequent chunk starts 40 - overlap words
    after the previous chunk's starting position.
    """
    if not isinstance(overlap, int) or not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("Overlap must be an integer between 0 and 39.")

    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if len(policy_files) != 5:
        raise ValueError(
            f"Expected 5 policy documents, found {len(policy_files)}."
        )

    chunks = []
    step = CHUNK_SIZE - overlap

    for file_path in policy_files:
        words = file_path.read_text(encoding="utf-8").split()

        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunks.append({
                "source": file_path.name,
                "text": " ".join(words[start:end]),
            })

            if end == len(words):
                break

            start += step

    return chunks


def build_index(overlap=DEFAULT_OVERLAP):
    """
    Create TF-IDF vectors for all policy chunks.

    Returns:
        chunks: Original chunk dictionaries.
        vectorizer: Fitted TF-IDF vectorizer.
        matrix: TF-IDF document matrix.
    """
    chunks = chunk_documents(overlap)

    if not chunks:
        raise ValueError("No policy chunks available for indexing.")

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform([
        chunk["text"] for chunk in chunks
    ])

    return chunks, vectorizer, matrix


def retrieve(query, k=3, index=None):
    """
    Retrieve the top-k policy chunks using cosine similarity.

    Returns a list containing source, score, and text.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if index is None:
        index = build_index()

    chunks, vectorizer, matrix = index

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: (-scores[i], i)
    )[:k]

    return [
        {
            "source": chunks[i]["source"],
            "score": float(scores[i]),
            "text": chunks[i]["text"],
        }
        for i in ranked_indices
    ]


if __name__ == "__main__":
    chunks = chunk_documents(overlap=10)

    print("Total chunks:", len(chunks))

    counts = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    print("Chunks per document:", counts)

    index = build_index(overlap=10)

    results = retrieve(
        "refund policy late delivery",
        k=3,
        index=index
    )

    print("\nTop 3 retrieval results:")

    for result in results:
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['score']:.4f}"
            f"\nText: {result['text']}"
        )
