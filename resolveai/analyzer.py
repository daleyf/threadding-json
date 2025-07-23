# analyzer.py
"""Tiny LLM-based summarizer."""

from __future__ import annotations
import os, json
from pathlib import Path
from uuid import uuid4
from typing import Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from common import Packet

load_dotenv()
THREAD_INDEX = os.getenv("THREAD_INDEX", "x")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="LLM Analyzer")

OUTPUT_DIR = Path("analyzed-outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_PROMPT_CHARS = 12_000   # keep room for system + completion


@app.post("/ingest")
async def ingest(packet: Packet):
    # Build a compact prompt
    texts: List[str] = [
        f"[{m.level}] {m.category}: {m.payload}" for m in packet.messages
    ]
    prompt = "\n".join(texts)[-MAX_PROMPT_CHARS:]  # tail-truncate

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system",
                 "content": "You are a senior engineer summarizing code & logs. "
                            "Return a concise overview plus any anomalies. in 2 sentences"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        summary = resp.choices[0].message.content.strip()
        usage = resp.usage.model_dump() if hasattr(resp, "usage") else {}
        out: Dict[str, Any] = {
            "agent_id": packet.agent_id,
            "summary": summary,
            "prompt_tokens": usage.get("prompt_tokens", None),
            "completion_tokens": usage.get("completion_tokens", None)
        }

        fname = f"{Path(packet.agent_id).stem}.json"
        with open(OUTPUT_DIR / fname, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"✅ wrote {fname}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ LLM error: {exc}")

    return {"status": "processed"}
