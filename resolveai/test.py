#!/usr/bin/env python3
"""Send ONE packet per file from ./repos/{size}/ to /packet."""
from __future__ import annotations
import argparse, os, sys, pathlib, datetime as dt, requests

SKIP_NAMES = {".DS_Store"}
SKIP_EXTS  = {
    ".png", ".jpg", ".jpeg", ".gif", ".zip", ".ico",
    ".exe", ".dll", ".so", ".mp3", ".mp4"
}

def should_skip(fname: str) -> bool:
    ext = pathlib.Path(fname).suffix.lower()
    return fname in SKIP_NAMES or ext in SKIP_EXTS

def send_file(host: str, agent_id: str, fpath: str) -> None:
    fname = os.path.basename(fpath)

    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    packet = {
        "agent_id": fname,
        "messages": [
            {
                "ts": dt.datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "category": "default",
                "payload": {
                    "text": content,
                    "filename": fname
                }
            }
        ]
    }

    resp = requests.post(f"{host}/packet", json=packet)
    try:
        resp.raise_for_status()
        data = resp.json()


        print(f"[✓] {fname}")

    except requests.HTTPError:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        print(f"[✗] {fname}: {resp.status_code} – {detail}")

def main() -> None:
    ap = argparse.ArgumentParser("Send each file in its own packet")
    ap.add_argument("size", choices=["small", "med", "large"],
                    help="Folder under ./repos to process")
    ap.add_argument("--host", default="http://127.0.0.1:8000",
                    help="Distributor base URL")
    ap.add_argument("--agent-id", default="uploader-script",
                    help="agent_id to include in packets")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "repos", args.size)
    if not os.path.isdir(root):
        print(f"❌ Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if should_skip(fname):
                print(f"[→] {fname}: skipped")
                continue
            send_file(args.host, args.agent_id, os.path.join(dirpath, fname))
            total += 1

    print(f"✅ Finished – sent {total} file(s)")

if __name__ == "__main__":
    main()
