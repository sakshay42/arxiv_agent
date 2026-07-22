"""Stage 1 agent: intake -> arXiv research -> trends -> ideas -> human feedback -> refine.

    python agent.py                    # new session
    python agent.py --resume mysession # pick up where you left off
    python agent.py --verify           # read PDFs to confirm affiliations (slower)
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import date, timedelta
from pathlib import Path

import arxiv_search as ax
import llm
import prompts

SESSIONS = Path(__file__).parent / "sessions"
W = 88


# ------------------------------------------------------------------ output ---

def rule(title: str = "") -> None:
    if title:
        print(f"\n{'=' * W}\n{title}\n{'=' * W}")
    else:
        print("-" * W)


def wrap(text: str, indent: str = "") -> str:
    out = []
    for para in text.split("\n"):
        out.append(textwrap.fill(para, W, initial_indent=indent,
                                 subsequent_indent=indent) if para.strip() else "")
    return "\n".join(out)


def ask(q: str, default: str = "") -> str:
    val = input(f"\n{q}\n> ").strip()
    return val or default


# ----------------------------------------------------------------- session ---

def new_session() -> dict:
    return {"name": "", "intake": {}, "rounds": [], "memo": "",
            "papers": {}, "kept": []}


def save(state: dict) -> Path:
    SESSIONS.mkdir(exist_ok=True)
    p = SESSIONS / f"{state['name']}.json"
    p.write_text(json.dumps(state, indent=2))
    return p


def load(name: str) -> dict:
    return json.loads((SESSIONS / f"{name}.json").read_text())


# ------------------------------------------------------------------ stages ---

def stage_intake(state: dict) -> None:
    rule("STAGE 1/5  INTAKE")
    print("Seven short questions. Enter to skip any.")
    for key, question in prompts.INTAKE_QUESTIONS:
        state["intake"][key] = ask(question)
    save(state)


def stage_research(state: dict, since: str, verbose: bool) -> tuple[str, list[ax.Paper]]:
    rule("STAGE 2/5  SEARCHING ARXIV")
    found: dict[str, ax.Paper] = {}

    def search_arxiv(query: str, max_results: int = 30, since: str = since) -> str:
        papers = ax.search(query, max_results=min(max_results, 60), since=since)
        for p in papers:
            found.setdefault(p.id, p)
        if not papers:
            return "No results. Try broader terms or drop a category filter."
        papers.sort(key=lambda x: (-x.industry_score, x.published), reverse=False)
        return f"{len(papers)} results:\n\n" + "\n\n".join(
            p.brief(abstract_chars=420) for p in papers[:40]
        )

    report, _ = llm.tool_loop(
        system=prompts.RESEARCHER_SYSTEM,
        messages=[{"role": "user",
                   "content": prompts.intake_to_seed(state["intake"])}],
        tools=[prompts.SEARCH_TOOL],
        handlers={"search_arxiv": search_arxiv},
        verbose=verbose,
    )
    print(f"\nRetrieved {len(found)} unique papers.")
    return report, list(found.values())


def stage_verify(papers: list[ax.Paper], report: str, top_n: int) -> None:
    """Confirm affiliations for papers the report actually cited."""
    rule("STAGE 2b  VERIFYING AFFILIATIONS (reading PDF first pages)")
    cited = [p for p in papers if p.id in report]
    cited.sort(key=lambda p: -p.industry_score)
    for i, p in enumerate(cited[:top_n], 1):
        print(f"  [{i}/{min(top_n, len(cited))}] {p.id}", end=" ")
        ax.verify_affiliation(p)
        print(f"-> {', '.join(p.affiliations) or 'no company match'}")


def stage_ideas(state: dict, report: str, papers: list[ax.Paper], n: int) -> list[dict]:
    rule("STAGE 3/5  GENERATING PROJECT IDEAS")
    ranked = sorted(papers, key=lambda p: -p.industry_score)[:60]
    block = "\n\n".join(p.brief(abstract_chars=600) for p in ranked)
    data = llm.chat_json(
        system=prompts.IDEATION_SYSTEM,
        messages=[{"role": "user", "content": prompts.ideation_user(
            state["intake"], report, block, state["memo"], n)}],
    )
    return data.get("ideas", [])


def show_ideas(ideas: list[dict]) -> None:
    for i, idea in enumerate(ideas, 1):
        rule()
        print(f"[{i}] {idea.get('title', '?')}   ({idea.get('id', '')})")
        print(f"    {idea.get('effort', '?')} | {idea.get('compute', '?')}")
        rule()
        for label, key in [("", "one_liner"), ("Question", "question"),
                           ("Industry angle", "why_industry"),
                           ("Data", "data"), ("Baseline", "baseline"),
                           ("Risk", "risk"), ("Signal", "signal")]:
            if idea.get(key):
                head = f"{label}: " if label else ""
                print(wrap(f"{head}{idea[key]}", "  "))
                print()
        if idea.get("build_steps"):
            print("  Build:")
            for j, s in enumerate(idea["build_steps"], 1):
                print(wrap(f"{j}. {s}", "    "))
            print()
        if idea.get("papers"):
            print("  Papers: " + "  ".join(
                f"arxiv.org/abs/{pid}" for pid in idea["papers"]))


def stage_feedback(state: dict, ideas: list[dict]) -> bool:
    """Returns True if the user wants another round."""
    rule("STAGE 4/5  YOUR CALL")
    print("Which ideas do you want to keep?")
    print("  numbers (e.g. 1,3)  |  none  |  all")
    keep_raw = ask("Keep:", "none").lower()

    if keep_raw == "all":
        kept_idx = list(range(len(ideas)))
    elif keep_raw in ("none", ""):
        kept_idx = []
    else:
        kept_idx = [int(x) - 1 for x in keep_raw.replace(" ", "").split(",")
                    if x.strip().isdigit() and 0 < int(x) <= len(ideas)]

    kept = [ideas[i] for i in kept_idx]
    rejected = [ideas[i] for i in range(len(ideas)) if i not in kept_idx]

    print("\nWhat didn't work about the ones you passed on?")
    print("Be blunt and specific -- 'too close to my day job', 'needs a GPU I")
    print("don't have', 'the question is boring'. This steers the next round.")
    notes = ask("Notes:")

    rule("STAGE 5/5  UPDATING YOUR PROFILE")
    memo_data = llm.chat_json(
        system=prompts.FEEDBACK_SYSTEM,
        messages=[{"role": "user", "content": prompts.feedback_user(
            state["memo"],
            [{"id": i.get("id"), "title": i.get("title")} for i in ideas],
            [i.get("id") for i in kept],
            [i.get("id") for i in rejected],
            notes)}],
    )
    state["memo"] = memo_data.get("memo", state["memo"])
    state["kept"].extend(kept)
    state["rounds"].append({"ideas": ideas, "kept": [i.get("id") for i in kept],
                            "notes": notes, "memo_data": memo_data})
    save(state)

    print(wrap(state["memo"], "  "))
    if memo_data.get("hard_constraints"):
        print("\n  Hard constraints now on file:")
        for c in memo_data["hard_constraints"]:
            print(wrap(f"- {c}", "    "))

    again = ask("\nAnother round of ideas with this feedback applied? (y/n)", "n")
    return again.lower().startswith("y")


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", help="session name to continue")
    ap.add_argument("--ideas", type=int, default=5, help="ideas per round")
    ap.add_argument("--months", type=int, default=18, help="how far back to search")
    ap.add_argument("--verify", action="store_true",
                    help="read PDF page 1 to confirm affiliations (slower, accurate)")
    ap.add_argument("--verify-top", type=int, default=12)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    since = (date.today() - timedelta(days=30 * args.months)).isoformat()

    if args.resume:
        state = load(args.resume)
        rule(f"RESUMING: {state['name']}")
        print(wrap(f"Memo on file: {state['memo'] or '(none)'}", "  "))
    else:
        state = new_session()
        state["name"] = ask("Session name (used for the save file):",
                            f"session-{date.today()}")
        stage_intake(state)

    report, papers = stage_research(state, since, verbose=not args.quiet)
    if args.verify:
        stage_verify(papers, report, args.verify_top)

    rule("TREND REPORT")
    print(wrap(report))
    state["papers"] = {p.id: p.to_dict() for p in papers}
    save(state)

    while True:
        ideas = stage_ideas(state, report, papers, args.ideas)
        if not ideas:
            print("No ideas returned. Check your API key and try again.")
            break
        show_ideas(ideas)
        if not stage_feedback(state, ideas):
            break

    rule("DONE")
    print(f"  Session saved: {save(state)}")
    if state["kept"]:
        print(f"  Ideas you kept ({len(state['kept'])}):")
        for i in state["kept"]:
            print(f"    - {i.get('title')}")


if __name__ == "__main__":
    main()
