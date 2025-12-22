# Project Context

This is a Python project for generating slide images from text using LLMs and LangGraph. The codebase utilizes an agentic workflow with a planner and a slide generator, orchestrated by LangGraph. It is inspired by NotebookLM and aims to convert text documents into visual slide shows.

## Key Conventions

- Use `uv` for dependency management.
- Follow PEP 8 style guidelines.
- Use type hints for function parameters and returns (e.g., using `TypedDict` for LangGraph states).
- Use OpenRouter for LLM API access.
- Image generation uses specialized models (e.g., Gemini 2.5 Flash Image).

## Setup & Running

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Setup environment variables
cp .env.example .env

# Run the slide maker
python slide_maker.py
```

## Project Structure

- `slide_maker.py` - Core logic for the agentic slide making graph using LangGraph.
- `slide_parallel.py` - Parallelized version of the slide generation process.
- `demo/` - Contains sample input texts and generated output images.
- `texts/`, `texts-2/` - Directories for input text files.
- `slides_XX/` - Output directories for generated slide images.
- `pyproject.toml` - Project configuration and dependencies (managed by uv).

## Common Tasks

**Adding a new agent/node to the graph:**
1. Define a new function in `slide_maker.py`.
2. Update the `GraphState` TypedDict if necessary.
3. Add the node to the `StateGraph` and define its edges.

**Changing LLM models:**
- Update the `model` parameter in `generate_text` or `generate_image` calls within the node functions (e.g., `planner`, `generate_slide`).

## Important Notes

⚠️ NEVER commit API keys or secrets.
⚠️ Ensure `OPENROUTER_API_KEY` is set in the environment.
