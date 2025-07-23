"""
distributor.py — weighted round-robin, self-healing

Exposes:
  • POST /packet  – send pre-built Packet JSON
  • POST /upload  – send a file; we build a Packet for you
"""
from __future__ import annotations
import asyncio, json, csv, io, mimetypes, random, time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Dict

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger

from common import Packet, LogMessage

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ANALYZER_WEIGHTS: Dict[str, float] = {
    "http://analyzer1:8000": 0.6,
    "http://analyzer2:8000": 0.3,
    "http://analyzer3:8000": 0.1,
}
RETRY_COOLDOWN = 5                     # seconds an analyzer stays in “cool-off”
MAX_UPLOAD_BYTES = 1_000_000           # 1 MB cap
TEXT_EXTS = (".py", ".md", ".rst", ".txt", ".log",
             ".yaml", ".yml", ".ini", ".cfg", ".toml")

# --------------------------------------------------------------------------- #
# Weighted router
# --------------------------------------------------------------------------- #
class WeightedRoundRobinRouter:
    """
    Pre-computes a list where each URL is repeated in proportion to its weight.
    Selection is O(1) with an index pointer protected by a lock.
    """
    def __init__(self, weights: Dict[str, float]) -> None:
        seq: List[str] = []
        for url, w in weights.items():
            seq.extend([url] * max(1, int(w * 100)))
        random.shuffle(seq)
        self._seq = seq
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            url = self._seq[self._idx]
            self._idx = (self._idx + 1) % len(self._seq)
            return url

# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #
queue  = asyncio.Queue[Packet](maxsize=10_000)
router = WeightedRoundRobinRouter(ANALYZER_WEIGHTS)
_dead: Dict[str, float] = {}           # url -> retry_at epoch seconds

# --------------------------------------------------------------------------- #
# Lifespan: fire-and-forget dispatcher
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 launching dispatcher task")

    async def dispatcher() -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            while True:
                pkt = await queue.get()

                # pick a healthy analyzer (bounded attempts)
                for _ in range(len(router._seq)):
                    target = await router.next()
                    if _dead.get(target, 0) <= time.time():
                        break
                else:
                    # none available - small back-pressure
                    await asyncio.sleep(1)
                    await queue.put(pkt)
                    continue

                url = f"{target}/ingest"
                try:
                    await client.post(url, json=pkt.dict())
                    # mark healthy (might have just come back)
                    _dead.pop(target, None)
                    logger.info(f"✅ {pkt.agent_id} → {url}")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"❌ failed to send to {url}: {exc}")
                    _dead[target] = time.time() + RETRY_COOLDOWN
                    await queue.put(pkt)   # re-queue for another analyzer

    asyncio.create_task(dispatcher())
    yield

app = FastAPI(title="Weighted Distributor", lifespan=lifespan)

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.post("/packet")
async def receive(packet: Packet):
    await queue.put(packet)
    return {"status": "queued"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (>1 MB)")

    filename     = file.filename
    content_type = file.content_type or mimetypes.guess_type(filename)[0]
    logger.debug(f"UPLOAD: {filename}  ({content_type})")

    try:
        messages = _parse_file(content, filename, content_type)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Parse failed")
        raise HTTPException(400, f"Parsing failed: {exc}") from exc

    await queue.put(Packet(agent_id=filename, messages=messages))
    return {"status": "uploaded"}

# --------------------------------------------------------------------------- #
# Helpers (unchanged)
# --------------------------------------------------------------------------- #
def _parse_file(content: bytes, filename: str, content_type: str) -> List[LogMessage]:
    """
    * .json – list[LogMessage] or {"messages":[...]}
    * .csv  – rows become payloads
    * TEXT_EXTS – generic text/code files (chunked ≤2 kB)
    """
    now  = datetime.now(timezone.utc).isoformat()
    text = content.decode("utf-8", errors="ignore")

    # ---------- JSON ----------
    if filename.lower().endswith(".json") or "json" in (content_type or ""):
        data = json.loads(text)
        if isinstance(data, list):
            return [LogMessage(**item) for item in data]
        if isinstance(data, dict) and "messages" in data:
            return [LogMessage(**m) for m in data["messages"]]
        raise ValueError("Unrecognized JSON schema")

    # ---------- CSV ----------
    if filename.lower().endswith(".csv"):
        return [
            LogMessage(ts=now, level="INFO", category=filename, payload=row)
            for row in csv.DictReader(io.StringIO(text))
        ]

    # ---------- Generic text / source ----------
    if filename.lower().endswith(TEXT_EXTS):
        def make_msg(blob: str) -> LogMessage:
            return LogMessage(
                ts=now,
                level="INFO",
                category=filename,
                payload={"text": blob},
            )

        CHUNK = 2048
        if len(text) <= CHUNK:
            return [make_msg(text)]

        out, buf, size = [], [], 0
        for line in text.splitlines():
            if size + len(line) + 1 > CHUNK:
                out.append(make_msg("\n".join(buf)))
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if buf:
            out.append(make_msg("\n".join(buf)))
        return out

    # ---------- Fallback ----------
    raise ValueError(f"Unsupported file type: {filename}")
