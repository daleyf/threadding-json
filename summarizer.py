#!/usr/bin/env python3
"""
summarizer.py ─ lightweight Q&A helper
--------------------------------------
* Reads every *.json file in ./analyzed-outputs/
* Concatenates their raw content as the **system prompt**.
* If the prompt > 10 000 tokens, prints a memo and truncates to 10 000.
* Sends your question to an LLM (OpenAI by default) and prints the answer.

Usage
$ export OPENAI_API_KEY=sk-...
$ python summarizer.py "What were the top errors last hour?"
"""

from __future__ import annotations

import json, os, sys, glob, textwrap
from pathlib import Path

# ───────────────────────── Config ──────────────────────────
ANALYZED_DIR   = Path(__file__).parent / "analyzed-outputs"
MODEL          = os.getenv("MODEL_ID", "gpt-4o")
MAX_TOKENS     = 10_000          # absolute system-prompt cap

# ─────────────────── Token helpers ─────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def n_tokens(txt: str) -> int:                # noqa: ANN001
        return len(_enc.encode(txt))
    def trim_to_tokens(txt: str, limit: int) -> str:  # noqa: ANN001
        ids = _enc.encode(txt)[:limit]
        return _enc.decode(ids)
except ModuleNotFoundError:
    print("⚠️  tiktoken not installed; falling back to 4-char≈1-token rule")
    def n_tokens(txt: str) -> int:                # noqa: ANN001
        return max(1, len(txt) // 4)
    def trim_to_tokens(txt: str, limit: int) -> str:  # noqa: ANN001
        max_chars = limit * 4
        return txt[:max_chars]

# ─────────────────── Prompt builder ───────────────────────
def collect_summaries() -> str:
    pieces = []
    for fp in sorted(glob.glob(str(ANALYZED_DIR / "*.json"))):
        try:
            pieces.append(Path(fp).read_text())
        except Exception as e:                    # noqa: BLE001
            print(f"[WARN] skip {fp}: {e}", file=sys.stderr)

    prompt = "\n\n".join(pieces)
    tokens = n_tokens(prompt)
    if tokens > MAX_TOKENS:
        print(f"⚠️  Prompt is {tokens:,} tokens; truncating to {MAX_TOKENS:,}.")
        prompt = trim_to_tokens(prompt, MAX_TOKENS)
    return prompt

# ─────────────────── LLM call ─────────────────────────────
def ask_llm(system_prompt: str, question: str) -> str:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        sys.exit("OPENAI_API_KEY env var not set.")
    resp = openai.chat.completions.create(
        model     = MODEL,
        temperature = 0.0,
        messages  = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
    )
    usage = resp.usage
    print(f"🧮 prompt_tokens: {usage.prompt_tokens}, completion_tokens: {usage.completion_tokens}\n")
    return resp.choices[0].message.content.strip()


# ─────────────────── Main ─────────────────────────────────
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python summarizer.py \"Your question here\"")
    question = " ".join(sys.argv[1:])
    system_prompt = collect_summaries()
    answer = ask_llm(system_prompt, question)
    print(textwrap.fill(answer, width=88))

if __name__ == "__main__":
    main()
