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
| 2026-05-17 | `uv sync` | Set up the project virtual environment so Python dependencies stay isolated from the rest of the computer | Completed; dependencies were already resolved |
| 2026-05-17 | `uv run cli/keyword_search_cli.py tdf trapper` | Ran the current keyword-search CLI to check the reported count for `trapper` | Completed; output was `69` |
| 2026-05-17 | `uv run python -c 'import pickle; index=pickle.load(open("cache/index.pkl","rb")); docmap=pickle.load(open("cache/docmap.pkl","rb")); tf=pickle.load(open("cache/term_frequencies.pkl","rb")); token="trapper"; ids=sorted(index.get(token,set())); print("matching_doc_count", len(ids)); print("matching_doc_ids", ids[:20]); print("term_frequency_sum", sum(tf.get(i,{}).get(token,0) for i in ids)); print("per_doc_frequency", [(i, tf.get(i,{}).get(token,0)) for i in ids[:20]]); print("docmap_sample", docmap.get(ids[0]) if ids else None)'` | Inspected the cached inverted index to compare matching document count with total word occurrence count | Completed; `trapper` appears in 11 documents and 69 total occurrences |
| 2026-05-19 | `uv sync` | Set up and checked the project virtual environment before running Python code | Completed; dependencies were already installed |
| 2026-05-19 | `uv run cli/semantic_search_cli.py verify` | Tested the semantic-search CLI `verify` subcommand | First failed because `argparse` parsed arguments before the `verify` subcommand was registered |
| 2026-05-19 | `uv run cli/semantic_search_cli.py verify` | Re-tested the semantic-search CLI after fixing argument parsing | Failed again because `max_seq_length` was read from the wrapper object instead of the SentenceTransformer model |
| 2026-05-19 | `uv run cli/semantic_search_cli.py verify` | Verified the semantic-search model loads after both fixes | Completed; printed the loaded model and max sequence length `256` |
| 2026-05-23 | `uv sync` | Set up the project virtual environment before inspecting semantic chunked search | Completed; dependencies were already installed |
| 2026-05-23 | `uv run cli/semantic_search_cli.py search_chunked "superhero action movie" --limit 25` | Reproduced the chunked semantic search output for the superhero query | Completed; output did not include the expected titles in the top 25 |
| 2026-05-23 | `uv run python -c 'from search_utils import load_movies; movies=load_movies(); titles={"Kick-Ass","The Incredibles","Logan"}; print("first_five", [(i, m["id"], m["title"]) for i,m in enumerate(movies[:5])]); print("expected", [(i, m["id"], m["title"]) for i,m in enumerate(movies) if m["title"] in titles])'` | Tried to inspect movie list positions from the project root | Failed with `ModuleNotFoundError: No module named 'search_utils'`, showing that direct scripts need the `cli` folder on `PYTHONPATH` |
| 2026-05-23 | `PYTHONPATH=cli uv run python -c 'from search_utils import load_movies; movies=load_movies(); titles={"Kick-Ass","The Incredibles","Logan"}; print("first_five", [(i, m["id"], m["title"]) for i,m in enumerate(movies[:5])]); print("expected", [(i, m["id"], m["title"]) for i,m in enumerate(movies) if m["title"] in titles])'` | Compared Python list indexes with movie IDs for the expected titles | Completed; the expected movies have zero-based list indexes one less than their one-based IDs |
| 2026-05-23 | `PYTHONPATH=cli uv run python -c 'from search_utils import load_movies; movies=load_movies(); indexes=[1176,1177,1178,3313,3314,3315,4431,4432,4433]; print([(i, movies[i]["id"], movies[i]["title"]) for i in indexes])'` | Checked neighboring movies around the expected titles | Completed; several printed wrong titles are immediately before the expected titles in the movie list |

## Change Log

Record meaningful repository changes here.

| Date | Files | What Changed | Why |
| --- | --- | --- | --- |
| 2026-05-16 | `AGENTS.md`, `README.md` | Added agent instructions and created this learning log | To make future agent work tutor-like and easy to audit or clean up |
| 2026-05-16 | `data/course-rag-movies.json` | Added movie dataset for RAG training | Source data for building the search engine |
| 2026-05-17 | `README.md` | Logged the `uv` setup and CLI/index inspection commands | To keep the learning record current while verifying keyword-search behavior |
| 2026-05-19 | `cli/semantic_search_cli.py` | Moved `parser.parse_args()` after subcommand registration and removed an invalid nested parser call | So `argparse` recognizes `verify` as a valid subcommand before parsing command-line input |
| 2026-05-19 | `cli/lib/semantic_search.py` | Read `max_seq_length` from `ss.model` instead of `ss` and renamed local variables to normal lowercase names | The wrapper class owns the model, but the model owns the sequence-length setting |
| 2026-05-19 | `README.md` | Logged the semantic-search CLI debugging steps | To preserve what failed, what changed, and how the fix was verified |
| 2026-05-23 | `README.md` | Logged the chunked semantic search debugging commands and observations | To keep the learning record current while investigating the mapping issue |

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
