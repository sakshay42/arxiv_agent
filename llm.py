"""Thin Anthropic wrapper: plain calls, JSON calls, and a tool-use loop."""

from __future__ import annotations

import json
import os
import re

import anthropic

MODEL = os.environ.get("TS_AGENT_MODEL", "claude-sonnet-5")
_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def chat(system: str, messages: list[dict], max_tokens: int = 4000) -> str:
    """Single turn, text back."""
    r = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system, messages=messages
    )
    return "".join(b.text for b in r.content if b.type == "text")


def chat_json(system: str, messages: list[dict], max_tokens: int = 8000):
    """Single turn, parsed JSON back. Tolerates ```json fences."""
    raw = chat(system + "\n\nRespond with valid JSON only. No prose, no code fences.",
               messages, max_tokens)
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # Last resort: grab the outermost brace/bracket span
        m = re.search(r"[\{\[].*[\}\]]", txt, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def tool_loop(system: str, messages: list[dict], tools: list[dict],
              handlers: dict, max_turns: int = 12, max_tokens: int = 8000,
              verbose: bool = True) -> tuple[str, list[dict]]:
    """Run Claude until it stops calling tools.

    handlers maps tool_name -> callable(**tool_input) -> str
    Returns (final_text, full_message_history).
    """
    msgs = list(messages)
    for turn in range(max_turns):
        r = _client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            messages=msgs, tools=tools,
        )
        msgs.append({"role": "assistant", "content": r.content})

        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), msgs

        results = []
        for block in r.content:
            if block.type != "tool_use":
                continue
            if verbose:
                preview = json.dumps(block.input)[:120]
                print(f"  -> {block.name}({preview})")
            try:
                out = handlers[block.name](**block.input)
            except Exception as e:  # noqa: BLE001 -- surface errors to the model
                out = f"ERROR: {type(e).__name__}: {e}"
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(out)[:60000],
            })
        msgs.append({"role": "user", "content": results})

    return "(hit max_turns without a final answer)", msgs
