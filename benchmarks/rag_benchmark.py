import pandas as pd

from src.rag_engine import retrieve_documents, build_context
from src.ollama_client import evaluate_model
from src.evaluator import evaluate_accuracy

MODELS = ["qwen2.5:7b", "qwen3:8b"]

TESTS = [
    ("What is the internal quantum benchmark identifier?", "QLAB-7824"),
    ("What does QAOA stand for?", "Quantum Approximate Optimization Algorithm"),
    ("What does VQE stand for?", "Variational Quantum Eigensolver"),
    ("Which quantum SDK is mentioned?", "Qiskit"),
    ("What is the internal project password?", "UNKNOWN")
]

results = []

for model in MODELS:
    for question, expected in TESTS:
        documents = retrieve_documents(question)
        context = build_context(documents)

        prompt = (
            "Answer using only the context below. "
            "If the answer is not present, return UNKNOWN. "
            "Return only the answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        print(f"Testing {model}: {question}", flush=True)

        result = evaluate_model(prompt, model)
        score = evaluate_accuracy(result["response"], expected)

        results.append({
            **result,
            **score,
            "question": question,
            "sources": ", ".join(
                doc["source"] for doc in documents
            )
        })

df = pd.DataFrame(results)

df.to_csv(
    "reports/rag_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.groupby("model")["accuracy_score"].mean())
print("RAG benchmark completed.")