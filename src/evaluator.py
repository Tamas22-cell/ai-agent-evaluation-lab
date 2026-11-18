import re


def normalize(text):
    text = str(text).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".!?").strip()


def evaluate_accuracy(response, expected):
    response = normalize(response)
    expected = normalize(expected)

    exact_match = bool(expected) and response == expected

    keyword_match = bool(
        expected and re.search(
            r"\b" + re.escape(expected) + r"\b",
            response
        )
    )

    expanded_match = bool(
        expected and re.fullmatch(
            r"[a-z0-9-]+ stands for " + re.escape(expected),
            response
        )
    )

    content_match = exact_match or expanded_match

    return {
        "expected": expected,
        "exact_match": exact_match,
        "keyword_match": keyword_match,
        "content_match": content_match,
        "exact_score": int(exact_match) * 100,
        "accuracy_score": int(content_match) * 100
    }


if __name__ == "__main__":
    print(evaluate_accuracy("6000", "6000"))
    print(evaluate_accuracy("5000", "6000"))
    print(evaluate_accuracy("Qiskit.", "Qiskit"))
    print(evaluate_accuracy(
        "VQE stands for Variational Quantum Eigensolver.",
        "Variational Quantum Eigensolver"
    ))