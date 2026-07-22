"""arXiv retrieval + industry-origin signals.

Two things happen here:
  1. search()             -- hit the arXiv API, normalise results
  2. prescreen()          -- cheap keyword scoring for "did industry write this?"
  3. verify_affiliation() -- expensive ground truth: read page 1 of the PDF

The agent calls search() and prescreen() constantly; verify_affiliation() is
reserved for the final shortlist because it downloads a PDF per paper.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict

import arxiv

# --------------------------------------------------------------------------
# Industry signal vocabulary
# --------------------------------------------------------------------------

# Orgs whose names appear in affiliation blocks. Lowercase, matched as substrings
# against page-1 PDF text. Extend freely -- this list is the main lever you have
# for steering the agent toward the labs you care about.
BIG_TECH = [
    "google", "deepmind", "google research", "google cloud",
    "amazon", "aws", "amazon web services",
    "microsoft", "microsoft research", "msra",
    "meta ", "facebook", "meta ai", "fair",
    "apple", "nvidia", "ibm research", "oracle", "sap se",
    "salesforce", "adobe research", "netflix", "uber", "lyft",
    "airbnb", "linkedin", "spotify", "datadog", "snowflake",
    "databricks", "servicenow", "cloudflare", "shopify", "doordash",
    "alibaba", "ant group", "tencent", "baidu", "huawei", "bytedance",
    "samsung", "sony", "bosch", "siemens", "ge research", "schneider electric",
    "nixtla", "abacus.ai", "cohere", "openai", "anthropic",
]

FINANCE = [
    "jpmorgan", "j.p. morgan", "jp morgan", "goldman sachs", "morgan stanley",
    "bloomberg", "citadel", "two sigma", "jane street", "hudson river trading",
    "de shaw", "d. e. shaw", "man group", "man ahl", "worldquant", "point72",
    "aqr", "blackrock", "vanguard", "fidelity", "state street",
    "capital one", "american express", "visa inc", "mastercard", "paypal",
    "stripe", "block, inc", "barclays", "hsbc", "ubs ", "credit suisse",
    "deutsche bank", "bnp paribas", "societe generale", "nomura",
    "wells fargo", "bank of america", "citigroup", "royal bank of canada",
    "rbc capital", "santander", "ing bank", "mizuho",
]

# Abstract-level phrases that suggest a deployed system rather than a benchmark
# paper. These are the tell for applied industry work even when the affiliation
# is invisible.
DEPLOYMENT_PHRASES = [
    "in production", "production system", "deployed", "deployment",
    "a/b test", "online experiment", "live traffic", "serving",
    "at scale", "industrial", "real-world deployment", "our platform",
    "millions of", "billions of", "customers", "business impact",
    "latency budget", "inference cost", "throughput", "cost savings",
    "case study at", "lessons learned", "practical",
]

# Well-known industry-originated time-series models. Presence of these names is
# a strong hint the paper sits in the industry lineage even if it's a follow-up.
INDUSTRY_MODELS = [
    "chronos", "timesfm", "moirai", "timegpt", "lag-llama", "toto",
    "moment", "tirex", "timer", "granite", "tabpfn-ts", "sundial",
]


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    url: str
    pdf_url: str
    comment: str = ""
    journal_ref: str = ""
    industry_score: int = 0
    industry_hits: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def brief(self, abstract_chars: int = 500) -> str:
        """Compact form fed to the LLM -- keeps the context window sane."""
        aff = f" | affil: {', '.join(self.affiliations)}" if self.affiliations else ""
        hits = f" | signals: {', '.join(self.industry_hits[:5])}" if self.industry_hits else ""
        return (
            f"[{self.id}] {self.title} ({self.published})\n"
            f"  authors: {', '.join(self.authors[:5])}\n"
            f"  industry_score: {self.industry_score}{aff}{hits}\n"
            f"  abstract: {self.abstract[:abstract_chars]}"
        )


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

_client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)


def _norm_id(entry_id: str) -> str:
    return re.sub(r"v\d+$", "", entry_id.rstrip("/").split("/")[-1])


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def search(query: str, max_results: int = 40, since: str | None = None) -> list[Paper]:
    """Run one arXiv query. `query` uses arXiv boolean syntax, e.g.
    'abs:"time series" AND cat:cs.LG AND abs:forecasting'
    `since` is an inclusive YYYY-MM-DD floor on the submission date.
    """
    s = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    out: list[Paper] = []
    for r in _client.results(s):
        pid = _norm_id(r.entry_id)
        published = r.published.strftime("%Y-%m-%d") if r.published else ""
        if since and published and published < since:
            continue
        p = Paper(
            id=pid,
            title=_clean(r.title),
            abstract=_clean(r.summary),
            authors=[a.name for a in r.authors][:10],
            published=published,
            updated=r.updated.strftime("%Y-%m-%d") if r.updated else "",
            categories=list(r.categories),
            url=f"https://arxiv.org/abs/{pid}",
            pdf_url=r.pdf_url or f"https://arxiv.org/pdf/{pid}",
            comment=_clean(r.comment or ""),
            journal_ref=_clean(r.journal_ref or ""),
        )
        prescreen(p)
        out.append(p)
    return out


def search_many(queries: list[str], max_results: int = 40,
                since: str | None = None) -> list[Paper]:
    """Run several queries, dedupe by arXiv id, keep first occurrence."""
    seen: dict[str, Paper] = {}
    for q in queries:
        try:
            for p in search(q, max_results=max_results, since=since):
                seen.setdefault(p.id, p)
        except Exception as e:  # noqa: BLE001 -- one bad query must not abort the run
            print(f"  ! query failed: {q} ({type(e).__name__}: {e})")
    return list(seen.values())


# --------------------------------------------------------------------------
# Industry scoring
# --------------------------------------------------------------------------

def prescreen(p: Paper) -> Paper:
    """Cheap 0-6 industry-origin score from metadata only. No network calls.

    This is deliberately a *recall* filter, not a precision one -- it is there to
    rank candidates before the LLM and PDF stages, both of which cost money/time.
    """
    hay = f"{p.title} {p.abstract} {p.comment} {p.journal_ref}".lower()
    score, hits = 0, []

    for org in BIG_TECH + FINANCE:
        if org in hay:
            score += 2
            hits.append(f"org:{org.strip()}")
            break

    dep = [d for d in DEPLOYMENT_PHRASES if d in hay]
    if dep:
        score += min(2, len(dep))
        hits.extend(f"deploy:{d}" for d in dep[:3])

    mod = [m for m in INDUSTRY_MODELS if m in hay]
    if mod:
        score += 1
        hits.extend(f"model:{m}" for m in mod[:3])

    # Applied venues skew industrial
    if re.search(r"\b(kdd|cikm|wsdm|recsys|www conference|applied data science)\b", hay):
        score += 1
        hits.append("venue:applied")

    p.industry_score = min(score, 6)
    p.industry_hits = hits
    return p


def verify_affiliation(p: Paper, timeout: int = 20) -> Paper:
    """Ground truth: pull page 1 of the PDF and look for org names.

    Requires `pymupdf` and `requests`. Costs ~1-3s per paper, so call it only on
    the shortlist. Sets p.verified=True whether or not orgs were found, so you
    can distinguish "checked, none found" from "not checked".
    """
    try:
        import requests
        import pymupdf
    except ImportError:
        print("  ! verify_affiliation needs: pip install pymupdf requests")
        return p

    try:
        resp = requests.get(p.pdf_url, timeout=timeout,
                            headers={"User-Agent": "ts-idea-agent/0.1"})
        resp.raise_for_status()
        doc = pymupdf.open(stream=io.BytesIO(resp.content), filetype="pdf")
        head = doc[0].get_text()[:4000].lower()
        doc.close()
    except Exception as e:  # noqa: BLE001
        print(f"  ! PDF fetch failed for {p.id}: {type(e).__name__}")
        return p

    found = []
    for org in BIG_TECH + FINANCE:
        if org in head:
            found.append(org.strip())
    # Email domains are a strong secondary signal
    for dom in set(re.findall(r"@([a-z0-9.-]+\.[a-z]{2,})", head)):
        if not any(dom.endswith(edu) for edu in (".edu", ".ac.uk", ".edu.cn", ".ac.jp")):
            found.append(f"email:{dom}")

    p.affiliations = sorted(set(found))[:8]
    p.verified = True
    if p.affiliations:
        p.industry_score = min(p.industry_score + 3, 9)
    return p
