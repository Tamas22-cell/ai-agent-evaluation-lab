import pandas as pd
from src.ollama_client import evaluate_model
from src.evaluator import evaluate_accuracy

MODELS = ["qwen2.5:7b", "qwen3:8b"]

TESTS = [
    ("Calculate 125 * 48.", "6000"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many bits are in one byte?", "8"),
    ("What is the capital of Italy?", "Rome")
]

results = []

for model in MODELS:
    for question, expected in TESTS:
        prompt = question + " Return only the answer."

        print(f"Testing {model}: {question}", flush=True)

        result = evaluate_model(prompt, model)
        score = evaluate_accuracy(
            result["response"], expected
        )

        results.append({**result, **score})

df = pd.DataFrame(results)

df.to_csv(
    "reports/accuracy_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.groupby("model")["accuracy_score"].mean())
print("Accuracy benchmark completed.")