import pandas as pd
from src.tool_engine import evaluate_tool_call

MODELS = ["qwen2.5:7b", "qwen3:8b"]

TESTS = [
    (125, 48),
    (25, 16),
    (99, 12),
    (250, 40),
    (17, 23)
]

results = []

for model in MODELS:
    for a, b in TESTS:
        expected = a * b
        prompt = f"Use the multiply tool to calculate {a} * {b}."

        print(f"Testing {model}: {a} * {b}", flush=True)

        try:
            result = evaluate_tool_call(prompt, model)

            success = any(
                call["tool"] == "multiply"
                and call["result"] == expected
                for call in result["results"]
            )

            error = ""

        except Exception as exc:
            success = False
            error = str(exc)

        results.append({
            "model": model,
            "prompt": prompt,
            "expected": expected,
            "success": success,
            "tool_score": int(success) * 100,
            "error": error
        })

df = pd.DataFrame(results)

df.to_csv(
    "reports/tool_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.groupby("model")["tool_score"].mean())
print("Tool calling benchmark completed.")
