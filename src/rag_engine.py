import re
from pathlib import Path


def tokenize(text):
    return set(re.findall(r"\w+", text.casefold()))


def retrieve_documents(query, top_k=3):
    query_words = tokenize(query)
    results = []

    for path in Path("data").glob("*.txt"):
        content = path.read_text(encoding="utf-8")
        document_words = tokenize(content)

        score = len(query_words & document_words)

        if score > 0:
            results.append({
                "source": path.name,
                "content": content,
                "score": score
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


def build_context(documents):
    return "\n\n".join(
        f"Source: {doc['source']}\n{doc['content']}"
        for doc in documents
    )