# MiniToolAgent

A lightweight ReAct agent framework for tool-augmented reasoning, supporting knowledge retrieval and mathematical problem-solving tasks.

## Features

- **ReAct loop**: Thought → Action → Observation cycle with configurable max steps
- **Reflection**: Automatic self-correction on parse errors, tool failures, and repeated actions
- **Built-in tools**: Wikipedia search (with optional Brave Search fallback) and sandboxed Python REPL
- **Resumable evaluation**: Checkpoints allow interrupted runs to continue from where they left off
- **Extensible**: Add custom tools by subclassing `Tool`; compose pipelines with `MiniToolAgentPipeline`

## Project Structure

```
MiniToolAgent/
├── minitoolagent/
│   ├── agent.py          # ReActAgent, ReflectionAgent
│   ├── pipeline.py       # MiniToolAgentPipeline (factory methods)
│   ├── evaluate.py       # Dataset evaluation loop
│   ├── models.py         # ChatModel (OpenAI-compatible API)
│   ├── memory.py         # AgentTrajectory, ActionStep
│   ├── parsing.py        # LLM response parser
│   ├── prompts.py        # System and task prompt templates
│   ├── config.py         # Config dataclass (loaded from llm.yaml)
│   └── tools/
│       ├── base.py       # Tool base class and ToolRegistry
│       ├── search.py     # WikipediaSearchTool
│       ├── python_repl.py# PythonREPLTool (sandboxed subprocess)
│       └── code_tool.py  # CodeTool (LLM-generated code execution)
├── data/
│   └── raw/
│       └── hotpotqa200.jsonl
├── llm.yaml.example      # Config template (copy to llm.yaml and fill in)
└── scripts/
```

## Setup

**1. Install dependencies**

```bash
pip install requests pyyaml sympy numpy scipy
```

**2. Configure the model**

Copy the example config and fill in your API credentials:

```bash
cp llm.yaml.example llm.yaml
```

`llm.yaml` fields:

```yaml
model_name: Qwen/Qwen2.5-72B-Instruct   # any OpenAI-compatible model
api_key: <your-api-key>
base_url: https://api.siliconflow.cn/v1  # or any OpenAI-compatible endpoint
max_tokens: 4096
temperature: 0.0
timeout: 120
brave_api_key: ""                         # optional: Brave Search API key
```

## Datasets

HotpotQA (200 samples) is already included in `data/raw/hotpotqa200.jsonl`. The math datasets (AIME 2025, HMMT Feb 2025) need to be downloaded and preprocessed first.

**1. Download from Hugging Face**

```bash
pip install datasets
python data/download_datasets.py
```

This saves the raw datasets to `data/raw/aime25/` and `data/raw/hmmt_feb_2025/`. The script uses `hf-mirror.com` automatically, so it works without a VPN.

**2. Preprocess into JSONL**

```bash
python data/preprocess_datasets.py
```

This produces:

- `data/processed/aime25.jsonl`
- `data/processed/hmmt_feb_2025.jsonl`

After preprocessing you can run any math evaluation script.

## Quick Start

All evaluation scripts live in `scripts/`. Common flags shared by every script:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `llm.yaml` | Path to model config |
| `--limit N` | full dataset | Evaluate only the first N items |
| `--run-name NAME` | auto | Experiment subfolder under `logs/` |
| `--no-reflection` | off | Disable self-correction |
| `--no-resume` | off | Ignore checkpoints, rerun from scratch |
| `--item-retries N` | 2 | Retry a failed item N extra times |

**HotpotQA — multi-hop QA with Wikipedia search**

```bash
python scripts/run_hotpotqa.py --limit 10
python scripts/run_hotpotqa.py --limit 50 --run-name qwen72b --max-steps 8
```

**Math (Python REPL) — AIME 2025 / HMMT**

```bash
python scripts/run_math.py --dataset aime25 --limit 10
python scripts/run_math.py --dataset hmmt --run-name qwen72b --max-steps 12
```

**Math v2 — pre-search math terms, then code**

```bash
python scripts/run_math_v2.py --dataset aime25 --limit 10
python scripts/run_math_v2.py --dataset aime25 --max-search-terms 3 --run-name qwen72b
```

**Code Math — LLM-generated code via CodeTool**

```bash
python scripts/run_code.py --dataset aime25 --limit 10
python scripts/run_code.py --dataset hmmt --run-name qwen72b_code
```

**Reflection — ReflectionAgent on one or all datasets**

```bash
# Single dataset
python scripts/run_ablation.py --dataset hotpotqa --limit 20

# All datasets in sequence
python scripts/run_ablation.py --dataset all --run-name reflection_exp
```

## Custom Tools

```python
from minitoolagent.tools.base import Tool, ToolParameter

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful."
    parameters = [
        ToolParameter(name="input", type="string", description="Input text.", required=True),
    ]

    def execute(self, input: str = "", **kwargs) -> str:
        return f"Result for: {input}"
```

## Available Pipeline Modes

| Mode | Agent | Tools | Dataset |
|------|-------|-------|---------|
| `hotpotqa` | ReActAgent | Wikipedia search | HotpotQA |
| `math` | ReActAgent | Python REPL | AIME 2025 / HMMT |
| `code_math` | ReActAgent | CodeTool | AIME 2025 / HMMT |
| `math_v2` | MathSearchThenCodeAgent | Wikipedia + Python REPL | AIME 2025 / HMMT |
| `reflection` | ReflectionAgent | task-dependent | HotpotQA / Math |

## Evaluation Output

Each run produces under the configured `output_dir`:

- `results.jsonl` — per-item predictions and correctness
- `summary.json` — accuracy, avg steps, token usage, timing
- `trajectory_<idx>.json` — full thought/action/observation trace per item
