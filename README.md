# RAG Search Engine Learning Exercise

This repository is a learning exercise for building and understanding a RAG search engine.

RAG means retrieval-augmented generation. In a RAG system, an application retrieves relevant information from a document collection and gives that information to a language model so the model can answer with better context.

The goal of this repository is not only to build code, but also to keep a clear record of what was learned, what changed, and what can be deleted when the exercise is complete.

## Learning Expectations

Agents working in this repository should respond like tutors. They should explain concepts, teach the reasoning behind changes, and document meaningful steps in this README.

## Project State

- Project name: `rag-search-engine`
- Language: Python
- Current package metadata: `pyproject.toml`
- Current entry point: `main.py`
- Current behavior: prints `Hello from rag-search-engine!`

## Command Log

Record commands here as the exercise progresses.

| Date | Command | Purpose | Result |
| --- | --- | --- | --- |
| 2026-05-16 | `brew install uv` | Installed `uv` through Homebrew so the project can use a modern Python package and virtual environment workflow | Completed before this log entry was added |
| 2026-05-16 | `mkdir -p data && cp ~/Desktop/course-rag-movies.json data/` | Copied movie dataset from Desktop into project `data/` folder | Completed; file is 25MB JSON |

## Change Log

Record meaningful repository changes here.

| Date | Files | What Changed | Why |
| --- | --- | --- | --- |
| 2026-05-16 | `AGENTS.md`, `README.md` | Added agent instructions and created this learning log | To make future agent work tutor-like and easy to audit or clean up |
| 2026-05-16 | `data/course-rag-movies.json` | Added movie dataset for RAG training | Source data for building the search engine |

## Cleanup Notes

Use this section to track anything that should be deleted at the end of the learning exercise.

- `data/course-rag-movies.json` (25MB) — copy from Desktop; can be removed or re-copied later

## Suggested Learning Path

1. Define the smallest useful RAG search engine behavior.
2. Add a small sample document collection.
3. Implement document loading and text chunking.
4. Add embeddings or a simple retrieval baseline.
5. Connect retrieval results to answer generation.
6. Add tests or verification commands for each step.
7. Review this README and remove exercise-only files when finished.
