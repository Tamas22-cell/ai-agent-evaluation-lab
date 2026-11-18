# AI Agent Evaluation Lab

An AI engineering platform for evaluating local
large language models, performance, accuracy,
RAG capabilities, and tool calling.

## Features

- Local LLM inference with Ollama
- NVIDIA GPU acceleration
- Model performance benchmarking
- Automated accuracy evaluation
- RAG evaluation
- Tool calling evaluation
- Interactive Streamlit dashboard
- CSV report export
- Automated pytest unit tests
- GitHub Actions CI configuration

## Technology Stack

- Python 3.14
- Ollama
- Qwen2.5 7B
- Qwen3 8B
- Streamlit
- Pandas
- Pytest
- GitHub Actions

## Hardware Environment

- AMD Ryzen 7 5700X3D
- NVIDIA RTX 3060 12 GB
- 32 GB RAM

## Project Structure

```text
ai-agent-evaluation-lab/
├── .github/workflows/
├── benchmarks/
├── data/
├── reports/
├── src/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install Ollama and download the required models:

```bash
ollama pull qwen2.5:7b
ollama pull qwen3:8b
```

## Run Benchmarks

```bash
python -m benchmarks.model_benchmark
python -m benchmarks.accuracy_benchmark
python -m benchmarks.rag_benchmark
python -m benchmarks.tool_benchmark
```

## Launch Dashboard

```bash
python -m streamlit run app.py
```

Dashboard: http://localhost:8501

## Automated Testing

```bash
python -m pytest -v -p no:cacheprovider
```

## Initial Results

| Model | Accuracy |
|---|---:|
| Qwen2.5 7B | 100% |
| Qwen3 8B | 100% |

Unit tests: 7 passed.

Results represent the initial limited test suite
and do not establish general model reliability.

## Roadmap

- Semantic evaluation
- Hallucination detection
- Advanced RAG benchmarking
- AI security evaluation
- Cloud LLM comparison
- Token cost analysis
- n8n workflow integration
- Automated evaluation pipelines

## Project Status

Active development.
GitHub Actions CI validation pending.

## Author

Tamas Nemeth