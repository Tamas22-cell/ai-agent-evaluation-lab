import ollama


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


TOOLS = {
    "multiply": multiply
}


def evaluate_tool_call(prompt, model="qwen3:8b"):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        tools=[multiply]
    )

    results = []

    for call in response.message.tool_calls or []:
        name = call.function.name
        arguments = call.function.arguments

        if name in TOOLS:
            result = TOOLS[name](**arguments)

            results.append({
                "tool": name,
                "arguments": arguments,
                "result": result
            })

    return {
        "model": model,
        "prompt": prompt,
        "tool_called": bool(results),
        "results": results
    }


if __name__ == "__main__":
    print(evaluate_tool_call(
        "Use the multiply tool to calculate 125 * 48."
    ))