# LLM Memory Evaluation Pipeline

A modular pipeline for evaluating different memory systems with Large Language Models.

## 🏗️ Architecture

The pipeline consists of modular components that can be mixed and matched:

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│   Dataset   │───▶│  Memory+LLM  │───▶│ Evaluator  │
│   Reader    │    │    System    │    │            │
└─────────────┘    └──────────────┘    └────────────┘
```

### Components

- **Core**: Abstract interfaces and pipeline orchestrator
- **Readers**: Dataset loaders (LongMemEval, custom formats)
- **Backends**: LLM providers (Ollama, OpenAI)
- **Memory Systems**: Different memory strategies (Full Context, Mem0)
- **Evaluators**: Answer evaluation (LLM Judge, F1 Score)
- **Utils**: Configuration, logging, output handling

## 📦 Installation

### Option 1: Minimal Installation (Core Only)

```bash
cd work/pipeline
pip install -r requirements.txt
```

### Option 2: Install Specific Components

```bash
# For Ollama backend
pip install -r requirements/ollama.txt

# For OpenAI backend
pip install -r requirements/openai.txt

# For Mem0 memory system
pip install -r requirements/mem0.txt

# For evaluators
pip install -r requirements/evaluators.txt
```

### Option 3: Full Installation (Everything)

```bash
pip install -r requirements/all.txt
```

### Option 4: Development Setup

```bash
pip install -r requirements/all.txt
pip install -r requirements/dev.txt
```

## 🚀 Quick Start

```python
from pipeline.core import EvaluationPipeline, PipelineConfig
from pipeline.readers.longmemeval import LongMemEvalReader
from pipeline.backends.ollama import OllamaBackend
from pipeline.memory_systems.full_context import FullContextMemorySystem
from pipeline.evaluators.llm_judge import LLMJudgeEvaluator

# Configure components
config = PipelineConfig(
    experiment_name="baseline_test",
    dataset_config={"name": "longmemeval"},
    memory_system_config={"type": "full_context"},
    llm_config={"model": "qwen2.5:32b"},
    evaluation_config={"judge_model": "gpt-4o-mini"},
    output_config={"output_dir": "results"}
)

# Initialize components
reader = LongMemEvalReader()
llm_backend = OllamaBackend()
memory_system = FullContextMemorySystem(llm_backend)
judge_backend = OpenAIBackend()
evaluator = LLMJudgeEvaluator(judge_backend)

# Create and run pipeline
pipeline = EvaluationPipeline(
    reader=reader,
    memory_system=memory_system,
    evaluators=[evaluator],
    config=config
)

results = pipeline.run("path/to/dataset.json")
```

## 📂 Project Structure

```
pipeline/
├── core/                  # Core interfaces and pipeline
│   ├── interfaces.py      # Abstract base classes
│   ├── models.py          # Data models
│   └── pipeline.py        # Main orchestrator
├── readers/               # Dataset readers
│   └── longmemeval.py     # LongMemEval format
├── backends/              # LLM backends
│   ├── ollama.py          # Ollama local models
│   └── openai.py          # OpenAI API
├── memory_systems/        # Memory strategies
│   ├── full_context.py    # Baseline (full context)
│   └── mem0_system.py     # Mem0 integration
├── evaluators/            # Evaluation methods
│   ├── llm_judge.py       # LLM-as-judge
│   └── f1_score.py        # Token-level F1
├── utils/                 # Utilities
│   ├── config.py          # Configuration management
│   ├── output.py          # Output handling
│   └── logging.py         # Logging utilities
└── requirements/          # Modular dependencies
    ├── core.txt           # Essential deps
    ├── ollama.txt         # Ollama backend
    ├── openai.txt         # OpenAI backend
    ├── mem0.txt           # Mem0 system
    ├── evaluators.txt     # Evaluators
    ├── dev.txt            # Development tools
    └── all.txt            # Everything
```

## 🔧 Adding New Components

### Add a New Memory System

1. Create file in `memory_systems/`
2. Inherit from `MemorySystem` interface
3. Implement required methods
4. Add dependencies to `requirements/` if needed

### Add a New Evaluator

1. Create file in `evaluators/`
2. Inherit from `Evaluator` interface
3. Implement evaluation logic
4. Add to pipeline evaluators list

### Add a New Backend

1. Create file in `backends/`
2. Inherit from `LLMBackend` interface
3. Implement generation method
4. Create requirements file if needed

## 📊 Development Status

- ✅ Core interfaces and models
- ✅ Pipeline orchestrator
- 📝 Component implementations (templates ready)
- ⏳ Testing suite
- ⏳ Documentation

## 📝 License

[Your License]

## 👤 Author

Javier - TFG: Impact of Memory/Long-Memory in Large Language Models
Universidad de Murcia
