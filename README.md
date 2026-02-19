# Bio-agent-connector

A lightweight Python framework for orchestrating [Biomni](https://github.com/snap-stanford/Biomni) A1 agents in chains and parallel workflows, with execution tracking and Word report generation.

## What it does

- **`BiomniAgentWrapper`** — wraps any Biomni `A1` agent to add per-call timing, history tracking, and async support
- **`BiomniAgentConnector`** — connects multiple wrapped agents into sequential chains or parallel pools, with optional output transformation between steps
- **`agent_logger.py`** — generates formatted Word (`.docx`) reports from chain execution results

## Project layout

```
Bio-agent-connector/
├── biomni_connector.py        # Core wrapper and connector classes
├── agent_logger.py            # Word report generation utilities
├── examples/
│   └── biomni_connector_example.ipynb   # End-to-end usage example
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Installation

### Requirements

- Python 3.10 or higher
- A valid [Anthropic API key](https://console.anthropic.com/) (Biomni uses Claude as its LLM)
- Conda or a Python virtual environment

### Step 1 — Create a clean environment

```bash
conda create -n bio-agent python=3.11 -y
conda activate bio-agent
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, Biomni will automatically download its data lake (~11 GB) to `./data/`.
> This is a one-time download. The `data/` folder is git-ignored.

### Step 3 — Configure your API key

```bash
cp .env.example .env      # Mac/Linux
copy .env.example .env    # Windows
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick start

```python
from dotenv import load_dotenv
load_dotenv()

from biomni.agent import A1
from biomni_connector import BiomniAgentConnector, extract_solution

# Initialize a Biomni agent
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Create connector and register agents
connector = BiomniAgentConnector()
connector.add_agent(agent, name="analyzer",  description="Analyzes biological data",  color="#e74c3c")
connector.add_agent(agent, name="summarizer", description="Summarizes findings",        color="#f39c12")
connector.add_agent(agent, name="writer",     description="Writes a narrative report",  color="#27ae60")

# Connect agents in a chain with output transformations
connector.connect("analyzer",  "summarizer",
    transform_func=lambda x: f"Summarize this analysis: {extract_solution(x)}")
connector.connect("summarizer", "writer",
    transform_func=lambda x: f"Write an engaging report from: {extract_solution(x)}")

connector.visualize_network()

# Run the chain (use await in a Jupyter notebook or async context)
results = await connector.execute_chain(
    start_agent="analyzer",
    initial_prompt="What pathway changes occur in liver cancer?"
)
```

### Generate a Word report

```python
from agent_logger import log_agent_results_readable

log_agent_results_readable(results, "my_report.docx", title="Liver Cancer Analysis")
```

---

## Running in parallel

```python
results = await connector.execute_parallel({
    "analyzer":  "What are the key mutations in LUAD?",
    "summarizer": "Summarize immunotherapy resistance mechanisms in TNBC",
})
```

---

## Tech stack

| Component | Library |
|---|---|
| Bio AI agent | [Biomni](https://github.com/snap-stanford/Biomni) (`A1`) |
| LLM backend | Anthropic Claude via `anthropic` |
| Bioinformatics | `biopython`, `rpy2` |
| Data | `pandas`, `numpy` |
| Reports | `python-docx` |
| Visualization | `matplotlib`, `seaborn` |
