# ResearchPilot

[![CI](https://github.com/nilswern/research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/nilswern/research-agent/actions/workflows/ci.yml)

An autonomous research agent built with **LangGraph**, **Google Gemini** and a local
**ChromaDB** vector store. Give it a question; it plans, decides which tools to use,
researches across several rounds, remembers what it found, validates its own
citations, and writes a cited Markdown report.

Use it from a local browser UI or from the command line — both drive the same
graph and share the same memory.

Everything runs on free tiers: Gemini via Google AI Studio, DuckDuckGo without an
API key, and local embeddings — no per-chunk API calls.

## Architecture

```mermaid
graph TD
    START([question]) --> plan[plan]
    plan --> agent[agent]
    agent -->|tool calls| tools[tools]
    agent -->|no tool calls| synthesize[synthesize]
    tools -->|budget left| agent
    tools -->|budget spent| synthesize
    synthesize --> validate[validate]
    validate -->|issues found| repair[repair]
    repair --> validate
    validate -->|clean| END([report])
```

Regenerate the exact compiled graph with `python main.py --graph`.

Tool selection is made by the model, not hard-coded:

| Tool | Purpose |
| --- | --- |
| `search_memory` | Semantic search over material from this and previous runs |
| `google_search` | General web search (DuckDuckGo, no API key) |
| `wikipedia_search` | Encyclopedic background |
| `arxiv_search` | Scientific abstracts |
| `scrape_webpage` | Full text of one URL, skipped if recently cached |

### Memory

Everything retrieved is chunked, embedded locally and persisted in ChromaDB with
full provenance: `source_tool`, `url`, `title`, `author`, `published_date`,
`retrieved_at`, `query`, `chunk_index`, `content_hash`, `document_id`,
`content_type`, `language`. Chunks older than `CACHE_MAX_AGE_DAYS` count as stale;
`--purge` removes them. Re-indexing a known URL upserts instead of duplicating.

### Source integrity

The report is validated against the set of URLs the tools actually returned.
Fabricated URLs and citation markers without a reference entry trigger a repair
round. If problems survive the repair budget, they are printed as a warning instead
of being silently shipped.

## Design decisions

**A state graph instead of a ReAct loop.** The research budget has to be
enforceable, so the control flow is data, not prompt text: `route_after_tools`
stops the loop once `MAX_RESEARCH_STEPS` rounds are spent, the recursion limit is
derived from the budgets, and `python main.py --graph` prints the compiled graph.
Anyone reviewing the agent can read the edges instead of guessing what the model
will do next.

**The model picks the tools, the code picks the limits.** All five tools are
bound to the LLM and selection happens per question, preceded by a planning node
that sets the direction. Hard-coded routing ("if the question mentions a paper,
call arXiv") is more predictable but breaks on every question you did not
anticipate. The budget, not a rule set, is what keeps runs bounded.

**Local embeddings over an embedding API.** Every retrieved passage gets embedded,
so a per-chunk API call would dominate both cost and rate limits.
`multilingual-e5-small` runs on CPU, handles German and English questions, and
makes the memory usable without a network round trip. The trade-off is retrieval
quality below large hosted models and a one-time model download.

**Provenance on every chunk, not just the text.** Each chunk carries `url`,
`document_id`, `content_hash`, `retrieved_at` and the tool that found it. That
single decision buys three features: re-indexing a known URL upserts instead of
duplicating (`document_id` is the URL hash), the scraper can skip pages fetched
recently, and stale material can be purged by age.

**The report is validated against reality.** A research agent that invents a
plausible URL is worse than one that says nothing, so the report is checked
against the set of URLs the tools actually returned; fabricated links and
citation markers without a reference entry trigger a repair round. Surviving
issues are shown as a warning rather than silently shipped — an extra LLM call is
cheap compared to a confidently fabricated source.

**One event stream, two frontends.** `src/graph/runner.py` emits typed events
(`plan`, `status`, `token`, `report`, `final`); the CLI renders them with `rich`,
the web UI forwards them as Server-Sent Events. Progress reporting exists once,
so the two interfaces cannot drift apart.

**A UI with no build step.** The page is plain HTML, CSS and JavaScript served by
FastAPI — no bundler, no `node_modules`, no CDN, including a small hand-written
Markdown renderer. Cloning the repo and running one command has to be enough;
a frontend toolchain would be a second project to maintain.

**One run at a time.** The vector store is a single local process, so the server
rejects a second concurrent research instead of pretending to scale. Honest
limits beat corrupted state.

### Known limitations

- Single user by design: the server binds to localhost and has no authentication.
- DuckDuckGo throttles aggressively; heavy use produces empty search rounds.
- Cancelling a run stops it at the next event — an LLM or tool call already in
  flight still finishes.
- Retrieval is pure vector similarity: no re-ranking, no hybrid keyword search.
- The Markdown renderer covers what the reports use, not the full spec.

## Setup

```powershell
# Windows
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # set GOOGLE_API_KEY from https://aistudio.google.com/apikey
```

```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

The browser UI and the CLI do the same work — same graph, same memory, same
reports. Pick whichever you prefer.

### Browser UI

Without a terminal, double-click one of the launchers in the project root:

| Launcher | Behaviour |
| --- | --- |
| `ResearchPilot.vbs` | Starts hidden in the background, opens the browser |
| `ResearchPilot.cmd` | Same, but keeps a window that shows errors |

Both serve <http://localhost:8000>. Stop the server with the **Quit** button
in the top right of the page — closing the browser tab leaves it running.

From a terminal:

```bash
python main.py --web                        # opens the browser on port 8000
python main.py --web --port 8080 --no-browser
```

The page is served by a local FastAPI app: type a question and watch the plan,
the tool rounds and the validation appear live while the report streams in token
by token, then gets rendered as Markdown. Sources, unresolved validation issues
and the name of the saved file sit next to it, and earlier reports from
`reports/` can be reopened from the sidebar. The options panel sets the same
knobs as the CLI flags: rounds, results per search, repair attempts, language and
whether to save.

Only one run executes at a time; a second request is rejected with a clear
message instead of competing for the vector store. Nothing leaves the machine
except the calls the agent makes itself.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The UI itself |
| `POST /api/research` | Run a research, streamed as Server-Sent Events |
| `GET /api/stats` | Memory statistics and default settings |
| `GET /api/reports` · `GET /api/reports/{name}` | List and read saved reports |
| `POST /api/purge` | Delete stale chunks |
| `POST /api/shutdown` | Stop the server (what the *Quit* button calls) |

The UI covers every CLI flag except `--graph`, the Mermaid export for developers.

### CLI

```bash
python main.py "How do vector databases handle metadata filtering?"
python main.py "What is RAG?" --max-steps 3 --no-stream
python main.py --stats
python main.py --graph
```

| Flag | Effect |
| --- | --- |
| `--max-steps N` | Override research rounds |
| `--results N` | Results per search |
| `--repairs N` | Report repair attempts |
| `--language en\|de` | Report language |
| `--no-stream` | Render the report at once instead of streaming |
| `--no-save` | Skip writing to `reports/` |
| `--purge` | Delete stale chunks first |
| `--stats` | Show memory statistics and exit |
| `--graph` | Print the compiled graph as Mermaid and exit |
| `--verbose` | Show agent logs |
| `--web` | Start the browser UI instead of a single run |
| `--host` · `--port` | Address for `--web` (default `127.0.0.1:8000`) |
| `--no-browser` | Do not open a browser tab with `--web` |

## Configuration

All settings live in `.env` (copy `.env.example`): model names, `CHUNK_SIZE`,
`RETRIEVAL_TOP_K`, `MAX_RESEARCH_STEPS`, `MAX_RESULTS_PER_SEARCH`,
`MAX_REPORT_REPAIRS`, `CACHE_MAX_AGE_DAYS`, output language and paths. Every
value has a default; only `GOOGLE_API_KEY` is required.

## Project layout

```
main.py                 entry point
ResearchPilot.vbs/.cmd  double-click launchers for the browser UI
src/
├── agent/              prompts
├── graph/              state, nodes, builder, mermaid export, run event stream
├── tools/              the five LLM tools
├── database/           ChromaDB layer and store factory
├── services/           llm, embeddings, chunking, scraper, validation, reporting
├── models/             Pydantic data models
├── config/             settings, logging
├── web/                FastAPI app, uvicorn launcher, static UI
└── cli.py
tests/                  unit tests, no network calls
reports/                generated Markdown reports
data/chroma/            the vector store
```

Both frontends consume the same event stream from `src/graph/runner.py`, so the
CLI and the UI always report the same progress.

## Development

```bash
pip install -r requirements-dev.txt
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
```

These are exactly the steps GitHub Actions runs on every push and pull request
(`.github/workflows/ci.yml`, on Python 3.11 and 3.12). Tool settings — line
length, lint rules, mypy strictness, pytest paths — live in `pyproject.toml`.

## AI assistance

This project was built with the help of an AI coding assistant (Claude), and it
seems fair to say where the line runs.

The design is mine. Choosing LangGraph over a hand-rolled agent loop, separating
the research rounds from a dedicated validate/repair stage, keeping embeddings
local and the memory persistent, and attaching provenance to every chunk so that
the freshness cache and the source validation can build on it — those decisions,
and the trade-offs behind them in [Design decisions](#design-decisions), came
first and shaped everything else.

My focus throughout was the framework itself: learning to model an agent in
LangGraph as an explicit state graph — nodes, conditional edges, a bounded
research loop, a separate repair stage and streamed events — instead of treating
the orchestration as a black box. That is the part of this project I care about
most, and the part I am happiest to be questioned on.

The assistant helped me turn them into code: implementations written against my
specifications, the browser UI, test scaffolding, and the tooling setup around
ruff, mypy and CI. I reviewed, corrected and tested what came back, and dropped
what did not match the intent. 