# Time-Series Project Idea Agent — Stage 1

An agent that asks what you're looking for, searches arXiv live, filters toward
**industry** work over academic benchmark-chasing, reports the trends, proposes
detailed projects, then takes your feedback and tries again.

## Quickstart (5 minutes)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py
```

That's it. It will ask you seven questions, then run.

Add `--verify` to have it download the first page of each shortlisted PDF and
confirm the affiliations rather than guessing from the abstract. Costs about
30 extra seconds.

```bash
python agent.py --verify --ideas 6 --months 12
python agent.py --resume my-session-name
```

## The loop

```
  1. INTAKE      7 questions: goal, domain, task, level, compute, time, avoid
        |
  2. RESEARCH    Claude writes its own arXiv queries and calls search_arxiv
        |        5-8 times, reformulating when results are thin
        |
  2b. VERIFY     (--verify) reads PDF page 1, regexes for company names
        |        and non-.edu email domains
        |
  3. TRENDS      Named trends + gaps + a reading list, all grounded in
        |        papers actually retrieved
        |
  4. IDEAS       N projects, each with a falsifiable question, dataset,
        |        baseline, build steps, risk, and effort estimate
        |
  5. FEEDBACK    You keep / reject / explain  <-- the human checkpoint
        |
        +------> memo updated, loop back to 4 (or 2 on a fresh run)
```

Everything persists to `sessions/<name>.json`, so the feedback memo survives
across runs. Round 3 knows what you hated in round 1.

## How the industry filter works

arXiv's API does not expose author affiliations. Three layers compensate:

| Layer | Cost | What it does |
|---|---|---|
| `prescreen()` | free | Keyword scan of title+abstract for company names, deployment vocabulary ("in production", "A/B test", "latency budget", "serving"), industry-origin model names (Chronos, TimesFM, Moirai, Toto), and applied venues. Scores 0–6. |
| LLM judgement | ~$0.10/run | The researcher prompt tells Claude to weigh deployment evidence over the heuristic score, and to override it when the abstract clearly describes a production system. |
| `verify_affiliation()` | ~2s/paper | Downloads the PDF, reads page 1, matches org names and email domains. Only runs on the shortlist. This is the only layer that's actually ground truth. |

Tested offline: a Chronos-style deployment abstract scores 5, a generic
"novel attention variant, SOTA on ETT/Electricity/Traffic" abstract scores 0.

**Tuning knob:** the lists at the top of `arxiv_search.py` (`BIG_TECH`,
`FINANCE`, `DEPLOYMENT_PHRASES`, `INDUSTRY_MODELS`). If you care about a
specific set of labs, add them there — that's the highest-leverage edit in the
repo.

## Files

| File | What's in it |
|---|---|
| `agent.py` | The 5-stage loop, CLI, session persistence |
| `arxiv_search.py` | arXiv API + industry scoring + PDF affiliation check |
| `prompts.py` | Every prompt. **Tune this file first.** |
| `llm.py` | Anthropic wrapper: `chat`, `chat_json`, `tool_loop` |

## Known limits (honest list)

1. **arXiv only.** Misses industry engineering blogs, KDD applied track, and
   internal tech reports — which is where a lot of real deployment writing lives.
2. **arXiv boolean search is keyword-based**, not semantic. Claude compensates by
   issuing many varied queries, but recall is imperfect.
3. **No dedup across rounds.** Round 2 can resurface a paper from round 1. Fine
   for now; the feedback memo handles idea-level repetition.
4. **`verify_affiliation` regexes plain text.** Two-column PDFs and image-only
   scans will miss.

## Model

Defaults to `claude-sonnet-5`. Override with `TS_AGENT_MODEL`:

```bash
export TS_AGENT_MODEL=claude-opus-4-8   # better synthesis, slower, pricier
```

## Rough cost per run

~15 arXiv calls (free) + one tool loop with ~40 abstracts in context + one
ideation call + one feedback call. Around $0.15–0.40 on Sonnet.
