"""Smoke-test the configured Gemini API without starting the factory app.

Usage:
    python scripts/test_gemini.py
    python scripts/test_gemini.py --model gemini-3.8-flash

The key is read from GEMINI_API_KEY or the local .env file and is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.8-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def load_local_env() -> None:
    """Load simple KEY=VALUE entries from .env without adding a dependency."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("FAIL: GEMINI_API_KEY is not set. Add it to .env or export it.")
        return 2

    url = f"{API_ROOT}/{args.model}:generateContent"
    payload = {
        "contents": [
            {"parts": [{"text": "Reply with exactly: Gemini API is working."}]}
        ],
        "generationConfig": {"temperature": 0},
    }

    try:
        response = httpx.post(
            url,
            json=payload,
            headers={"x-goog-api-key": key},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        print(f"FAIL: Gemini returned HTTP {exc.response.status_code}: {detail}")
        return 1
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not read a valid Gemini response: {type(exc).__name__}: {exc}")
        return 1

    usage = body.get("usageMetadata", {})
    print(f"PASS: {args.model} responded successfully.")
    print(f"Response: {text}")
    if usage:
        print(
            "Tokens: "
            f"prompt={usage.get('promptTokenCount', 0)}, "
            f"candidates={usage.get('candidatesTokenCount', 0)}, "
            f"total={usage.get('totalTokenCount', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
