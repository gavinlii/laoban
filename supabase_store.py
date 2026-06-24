"""
supabase_store.py — durable game-log storage in Supabase (Postgres), via the
PostgREST REST API using only the stdlib (no extra deployment deps).

Render's free tier has an ephemeral disk, so local game_*.json files are wiped on
every restart/redeploy. This mirrors each completed game into a Supabase table so
the data survives, and can be pulled to a local training machine on demand.

Config (env vars, set on the backend host; never commit these):
  SUPABASE_URL   e.g. https://abcdefgh.supabase.co
  SUPABASE_KEY   the service_role key (server-side secret; bypasses RLS)
  SUPABASE_TABLE optional, defaults to "games"

If the env vars are unset the store is simply disabled (no-op), so the app runs
fine locally without Supabase.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import List, Optional


class SupabaseGameStore:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, table: Optional[str] = None):
        self.url = (url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.key = key or os.getenv("SUPABASE_KEY", "")
        self.table = table or os.getenv("SUPABASE_TABLE", "games")
        self.enabled = bool(self.url and self.key)

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    # -- write one game ------------------------------------------------------
    def insert(self, record: dict, timeout: float = 10.0) -> bool:
        """Insert a single game record. Returns True on success. Best-effort:
        never raises, so a Supabase hiccup can't break gameplay."""
        if not self.enabled:
            return False
        row = {
            "session_id":  record.get("session_id"),
            "winner":      record.get("winner"),
            "bot_margin":  record.get("bot_margin"),
            "n_decisions": record.get("n_decisions"),
            "record":      record,
        }
        data = json.dumps(row).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/rest/v1/{self.table}",
            data=data, method="POST",
            headers=self._headers({"Prefer": "return=minimal"}),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status in (200, 201, 204)
        except Exception:
            return False

    # -- read all games ------------------------------------------------------
    def fetch_all(self, page: int = 500, timeout: float = 30.0) -> List[dict]:
        """Return every stored game record (the full per-game JSON), oldest first.
        Pages through PostgREST with limit/offset."""
        if not self.enabled:
            return []
        out: List[dict] = []
        offset = 0
        while True:
            url = (f"{self.url}/rest/v1/{self.table}"
                   f"?select=record&order=created.asc&limit={page}&offset={offset}")
            req = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    batch = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                print(f"[supabase] fetch error at offset {offset}: {exc}")
                break
            if not batch:
                break
            out.extend(r["record"] for r in batch if r.get("record"))
            if len(batch) < page:
                break
            offset += page
        return out

    def count(self, timeout: float = 15.0) -> int:
        """Number of stored games (uses PostgREST exact count header)."""
        if not self.enabled:
            return 0
        req = urllib.request.Request(
            f"{self.url}/rest/v1/{self.table}?select=session_id",
            headers=self._headers({"Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"}),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cr = resp.headers.get("Content-Range", "")  # e.g. "0-0/123"
                if "/" in cr:
                    tail = cr.split("/")[-1]
                    return int(tail) if tail.isdigit() else 0
        except Exception:
            pass
        return 0
