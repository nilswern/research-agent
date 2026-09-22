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
| `web_search` | General web search (DuckDuckGo, no API key) |
| `wikipedia_search` | Encyclopedic background |
| `arxiv_search` | Scientific abstracts |
| `scrape_webpage` | Full text of one URL, skipped if recently cached |

### Memory

Everything retrieved is chunked, embedded locally and persisted in ChromaDB with
full provenance: `source_tool`, `url`, `title`, `author`, `published_date`,
`retrieved_at` (plus a numeric `retrieved_at_ts` the freshness check filters on),
`query`, `chunk_index`, `content_hash`, `document_id`, `content_type`,
`language`. Chunks older than `CACHE_MAX_AGE_DAYS` count as stale;
`--purge` removes them. Re-indexing a known URL upserts instead of duplicating.

### Source integrity

The report is validated against the set of URLs the tools actually returned.
Fabricated URLs and citation markers without a reference entry trigger a repair
round. If problems survive the repair budget, they are printed as a warning instead
of being silently shipped.

A real URL is not the same as a supporting one, so there is a second, optional
check: with `FAITHFULNESS_CHECK=true`, every sentence carrying a citation is
compared against the stored chunks of exactly the source it cites, using the
embeddings already in ChromaDB. Sentences below `FAITHFULNESS_THRESHOLD` are
reported as unsupported claims and go to the same repair stage. It is off by
default because it costs one vector query per cited sentence, and it is a
similarity proxy: it catches a claim the source never discusses, not a subtly
wrong number.

## Design decisions

**A state graph instead of a ReAct loop.** The research budget has to be
enforceable, so the control flow is data, not prompt text: `route_after_tools`
stops the loop once `MAX_RESEARCH_STEPS` rounds are spent, the recursion limit is
derived from the budgets, and `python main.py --graph` prints the compiled graph.
Anyone reviewing the agent can read the edges instead of guessing what the model
will do next.

**The model picks the tools, the code picks the limits.** All five tools are
bound to the LLM, and a planning node sets the direction first. Hard-coded
routing ("if the question mentions a paper, call arXiv") is more predictable but
breaks on every question you did not anticipate. The budget, not a rule set, is
what keeps a run bounded.

**Local embeddings over an embedding API.** Every retrieved passage gets
embedded, so per-chunk API calls would dominate cost and rate limits.
`multilingual-e5-small` runs on CPU and handles German and English. The
trade-off is retrieval quality below large hosted models.

**Provenance on every chunk, not just the text.** The metadata listed under
[Memory](#memory) is what makes three features possible at once: re-indexing a
known URL upserts instead of duplicating (`document_id` is the URL hash), the
scraper can skip pages fetched recently, and stale material can be purged by age.

**The report is validated against reality.** A research agent that invents a
plausible URL is worse than one that says nothing, so the report is checked
against the set of URLs the tools actually returned; fabricated links and
citation markers without a reference entry trigger a repair round. Surviving
issues are shown as a warning rather than silently shipped — an extra LLM call is
cheap compared to a confidently fabricated source.

**One event stream, two frontends.** `src/graph/runner.py` emits typed events;
the CLI renders them with `rich`, the web UI forwards them as Server-Sent
Events. Progress reporting exists once, so the two cannot drift apart.

**A UI with no build step.** Plain HTML, CSS and JavaScript served by FastAPI —
no bundler, no `node_modules`, no CDN, down to a hand-written Markdown renderer.
Cloning the repo and running one command has to be enough; a frontend toolchain
would be a second project to maintain.

**One run at a time.** The vector store is a single local process, so the server
rejects a second concurrent research instead of pretending to scale. Honest
limits beat corrupted state.

**Fetched text is data, not instruction.** A research agent reads pages written
by strangers, so fetched text is fenced and machine-relevant data is read from
outside that fence. [Security](#security) describes the mechanism and its
limits.

### Known limitations

- Single user by design: the server binds to localhost and has no
  authentication; see [Security](#security) for the rest of the threat model.
- DuckDuckGo throttles aggressively; heavy use produces empty search rounds.
- Cancelling a run stops it at the next event — an LLM or tool call already in
  flight still finishes.
- Retrieval is pure vector similarity: no re-ranking, no hybrid keyword search.
- URL identity is normalised per host (arXiv `/abs/`, `/pdf/` and version
  suffixes collapse into one). The same paper mirrored on a different host still
  counts as a separate source.
- The faithfulness check measures similarity, not entailment: it flags claims a
  source never discusses, not ones it contradicts in detail.
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

## Example run

```bash
python main.py "How do vector databases handle metadata filtering?" --max-steps 3
```

The terminal shows the run as it happens (abridged):

```
╭─ Question ───────────────────────────────────────────╮
│ How do vector databases handle metadata filtering?   │
╰──────────────────────────────────────────────────────╯
Plan created
╭─ Research plan ──────────────────────────────────────╮
│ 1. Check memory  2. Compare pre- vs post-filtering   │
│ 3. Read one primary source in full                   │
╰──────────────────────────────────────────────────────╯
→ search_memory
  Round 1 · 2 sources
→ web_search, arxiv_search
  Round 2 · 11 sources
→ scrape_webpage
  Round 3 · 14 sources
Writing the report
───────────────────── Report ──────────────────────────
# How Vector Databases Handle Metadata Filtering
## Summary
Vector databases combine similarity search with attribute constraints ...
Validation: 1 issue(s), repairing
───────────────── Corrected report ────────────────────
...
Validation passed

Saved: reports/20260922-100711_how-do-vector-databases-handle-metadata-filtering.md
3 rounds · 14 sources · 1 repairs · 277 chunks in memory
```

That run needed one repair: the first draft did not pass validation, the
corrected one did. The complete report, with frontmatter and reference list, is
committed as [`examples/metadata-filtering.md`](examples/metadata-filtering.md).

The browser UI shows the same run: plan and activity log on the left, the report
streaming in on the right.

![The UI during a run: research plan and activity log on the left, the report
streaming in on the right](docs/ui-running.png)

![The same run finished: rendered report with inline citations and a validated
reference list, 4 rounds and 27 sources](docs/ui-report.png)

## Evaluation

Source discipline is a claim, so there is a harness that measures it. It runs
the agent over a fixed question set and records what actually happened:

```bash
python -m evals.run_eval                      # the whole set
python -m evals.run_eval --limit 3            # a quick pass
python -m evals.run_eval --category current
python -m evals.run_eval --delay 35           # pace it for a free-tier key
```

The question set in [`evals/questions.jsonl`](evals/questions.jsonl) has four
kinds of question, and the last kind is the interesting one:

| Category | What it probes |
| --- | --- |
| `factual` | One well-documented answer. The baseline. |
| `multi-hop` | Two or three sub-questions that need different sources. |
| `current` | Moving targets where the training data is wrong by now. |
| `hard-to-source` | Questions with no trustworthy answer, including one that is unanswerable by construction. Declining is the correct behaviour. |

Per question it records whether the first report passed validation without a
repair, how many repairs ran, how many fabricated URLs and dangling citations
survived, unsupported claims when the faithfulness check is on, research rounds,
sources, duration and token usage. Results are written to `evals/results/` as
JSON plus a Markdown table.

This makes real API calls, so it is never part of CI. The harness itself is
covered by offline tests using a fake model.

### Results

Run of 2026-09-22 from a cold start - the vector store was empty, so nothing
could come from memory. Three research rounds per question, faithfulness check
off. Raw data:
[`evals/results/20260922-full-run.json`](evals/results/20260922-full-run.json).

The model was `gemini-3.5-flash-lite`, not the `gemini-3.8-flash` default: the
free tier allows 15 requests per minute and a question costs seven or eight, so
the run was paced with `--delay 35`. Expect better answers and a slower, pricier
run on the default model.

| Category | Questions | 1st pass valid | Clean after repair | Fabricated URLs | Avg rounds | Avg sources |
| --- | --- | --- | --- | --- | --- | --- |
| factual | 3 | 2 | 2 | 0 | 2.7 | 13.7 |
| multi-hop | 4 | 3 | 4 | 0 | 3.0 | 15.8 |
| current | 3 | 3 | 3 | 0 | 3.0 | 18.3 |
| hard-to-source | 4 | 4 | 4 | 0 | 3.0 | 17.5 |
| **all** | **14** | **12 (86%)** | **13** | **0** | **2.9** | **16.4** |

The number that matters is the one in the middle of that last row: **no report
shipped a URL that no tool had returned**, across 14 questions and 229 collected
sources. Two reports failed validation on the first attempt and were repaired.
One report still had a defect afterwards - `fact-03`, a citation marker without
a matching entry in the reference list - and notably it came from the *easiest*
category, not from the unanswerable ones.

That last part is worth being careful about. An earlier run of the same set,
with a warm memory, produced 10/14 on the first pass and put three of its four
repairs in the `hard-to-source` group. Two runs, two different pictures: 14
questions is a small sample, and a single number from it is a weak claim. What
holds across both runs is the part the design is actually aimed at - zero
fabricated sources in the shipped report.

Cost per question: 16.5 seconds, about 13k input and 1.2k output tokens.

## Security

The agent reads web pages, and a web page can contain text aimed at the model
rather than at a reader. Three things follow from treating that as the default:

**Fetched text is fenced.** Page bodies, snippets, abstracts and stored passages
are wrapped in `<untrusted_content>` delimiters before they reach the model, with
any delimiter inside the payload neutralised first. The research and synthesis
prompts state that fenced text is evidence and never an instruction.

**Machine-relevant data is read outside the fence.** The allow-list of citable
URLs is built only from what a tool itself emitted - the URL it fetched, the URLs
a search API returned - never from inside fetched content. A page telling the
agent to cite `attacker.test` produces a URL the validation stage then rejects as
never returned by any tool.

**Control flow cannot be argued with.** Budgets and routing read counters and
message structure, not text: the round limit, the repair limit and the
validation result are computed in code, so no page can grant itself more rounds
or skip a check. Titles and author names are flattened to a single short line.

**The scraper cannot reach inward.** The model picks the URL, so `scrape_webpage`
resolves every host first and refuses loopback, private, link-local and reserved
addresses - including the cloud metadata endpoint `169.254.169.254`. Redirects
are followed manually, one hop at a time, and each hop is checked *before* it is
requested, so a public page cannot bounce the agent into `http://localhost:8000`.

`tests/test_security.py` covers each of these with an injected page or a forged
redirect.

### What this does not cover

- The model still *reads* the injected text. Fencing makes it unlikely to obey,
  not impossible; no prompt-level defence is a guarantee.
- A page can still mislead the agent with plausible false *content*. Injection
  defence is not fact-checking - that is what validation and the faithfulness
  check are for, and both have limits.
- The address check happens at resolution time. A host that resolves to a public
  address and then to a private one between check and connect (DNS rebinding)
  would slip through; closing that needs a pinned-IP transport.
- Any *public* URL is fair game: there is no domain allow-list and no sandbox
  around the tools.
- Nothing is rate-limited or quota-guarded beyond the research budget.

Treat this as a local single-user tool, not as something to expose to a network.

## Observability

Tracing needs no code and no dependency: LangChain reads these variables by
itself, and every node, tool call and model call of a run becomes one trace -
useful for seeing which round burned the budget. Unset, nothing changes.

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your key>
LANGSMITH_PROJECT=researchpilot
```

<!-- Add a trace screenshot as docs/trace.png and uncomment the next line. -->
<!-- ![A run in LangSmith](docs/trace.png) -->

## Configuration

All settings live in `.env` (copy `.env.example`): model names, `CHUNK_SIZE`,
`RETRIEVAL_TOP_K`, `MAX_RESEARCH_STEPS`, `MAX_RESULTS_PER_SEARCH`,
`MAX_REPORT_REPAIRS`, `CACHE_MAX_AGE_DAYS`, output language and paths. Every
value has a default; only `GOOGLE_API_KEY` is required.

Two optional switches are off unless you turn them on: `FAITHFULNESS_CHECK`
(plus `FAITHFULNESS_THRESHOLD`) adds the claim-support check described under
[Source integrity](#source-integrity), and the `LANGSMITH_*` variables enable
[tracing](#observability).

## Project layout

```
main.py                 entry point
ResearchPilot.vbs/.cmd  double-click launchers for the browser UI
src/
├── agent/              prompts
├── graph/              state, nodes, builder, mermaid export, run event stream
├── tools/              the five LLM tools
├── database/           ChromaDB layer and store factory
├── services/           llm, embeddings, chunking, scraper, reporting, validation,
│                       faithfulness, untrusted-content fencing
├── models/             Pydantic data models
├── config/             settings, logging
├── web/                FastAPI app, uvicorn launcher, static UI
└── cli.py
evals/                  question set and the measurement harness
tests/                  unit tests, no network calls
examples/               a generated report to look at
docs/                   screenshots for the README
reports/                generated Markdown reports
data/chroma/            the vector store
```

Both frontends consume the same event stream from `src/graph/runner.py`, so the
CLI and the UI always report the same progress.

## Development

```bash
pip install -r requirements-dev.txt
ruff check src tests evals
ruff format --check src tests evals
mypy src evals
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