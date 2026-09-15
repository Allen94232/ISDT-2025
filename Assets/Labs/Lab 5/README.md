# Lab 5 — LLM Puzzle Graphs

Lab 5 uses OpenAI vision and structured output to identify objects, generate puzzle graphs for predefined goals, save the results as JSON, and render graph visualizations with Graphviz.

## Requirements

- Python 3
- OpenAI Python SDK
- Pydantic
- Graphviz Python package
- Graphviz system executable available on `PATH`

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install openai pydantic graphviz
```

Install the Graphviz application separately if the `dot` executable is not already available.

## API Key

The script reads the key from `OPENAI_API_KEY`. Do not paste a key into the source file.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-key"
python "Lab 5.py"
```

bash:

```bash
export OPENAI_API_KEY="your-key"
python "Lab 5.py"
```

## Inputs and Outputs

| Path | Purpose |
| --- | --- |
| `images/` | Images used for object identification |
| `puzzle_graphs.json` | Structured puzzle-graph output |
| `puzzle_graphs_viz/` | Rendered graph images |

The script contains predefined goals and one hardcoded object set. When vision is enabled, a second object set is derived from images using `gpt-4o`.

## Run

From the Lab 5 directory:

```bash
python "Lab 5.py"
```

## Reproducibility Notes

- Model output can vary between runs despite fixed prompts and temperatures.
- Record the model name, prompt version, input images, and generated JSON when comparing results.
- Review generated graphs before treating them as valid puzzle solutions.
- Keep images free of private or sensitive information before sending them to an external API.
