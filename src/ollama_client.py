import ollama
import time


def evaluate_model(prompt, model="qwen3:8b"):
    start = time.perf_counter()

    result = ollama.generate(
        model=model,
        prompt=prompt
    )

    return {
        "model": model,
        "prompt": prompt,
        "response": result.response,
        "latency_seconds": round(
            time.perf_counter() - start, 3
        ),
        "tokens": result.eval_count
    }


if __name__ == "__main__":
    result = evaluate_model("What is quantum computing?")
    print(result)
    