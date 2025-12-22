# CLAUDE.md

## Build and Run Commands
- **Sync dependencies**: `uv sync`
- **Run core slide generator**: `python slide_maker.py`
- **Run parallel slide generator**: `python slide_parallel.py`

## Coding Style Guidelines
- **Language**: Python 3.13+
- **Style**: Follow PEP 8 guidelines.
- **Typing**: Use type hints for all function parameters and return values. Use `TypedDict` for LangGraph `GraphState`.
- **LLM Access**: Use OpenRouter API via the `generate_text` and `generate_image` helpers.
- **Graph Orchestration**: Use LangGraph for agentic workflows. Define nodes as functions and edges via `StateGraph`.
- **Dependencies**: Manage all dependencies via `pyproject.toml` and `uv`.
- **Naming**: Use snake_case for functions and variables, PascalCase for classes (e.g., `GraphState`).
- **Imports**: Organize imports into standard library, third-party (like `langgraph`, `PIL`), and local modules.

## Test
use `pytest`
