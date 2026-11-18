import pandas as pd
from src.ollama_client import evaluate_model

MODELS = [
    "qwen2.5:7b",
    "qwen3:8b"
]

PROMPTS = [
    "What is quantum computing?",
    "Explain blockchain consensus.",
    "What is Retrieval Augmented Generation?",
    "Calculate 125 * 48."
]

results = []

for model in MODELS:
    for prompt in PROMPTS:
        print(f"Testing {model}: {prompt}", flush=True)

        result = evaluate_model(prompt, model)
        results.append(result)

df = pd.DataFrame(results)

df.to_csv(
    "reports/benchmark_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df[["model", "latency_seconds", "tokens"]])
print("Benchmark completed.")