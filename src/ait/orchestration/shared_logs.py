"""Shared per-machine logs: each box publishes its own bot's structured events
to the repo so the other box can read them.

Why (2026-09-29): Prithvi's bot opened two condors on 09-28 and Ahmed's did
not, and nobody could say why — each box's `logs/` is git-ignored and never
leaves the machine. Ahmed's first attempt (rename `logs/` -> `logs_ahmed/` in
shared code and track it) was reverted by agreement: the folder name was a
code constant both boxes would share, a tracked folder the bot writes to
constantly breaks the deploy's `git merge --ff-only`, and the raw logs are
~200 MB/day (mostly IBKR's delayed-data prints) with a documented history of
leaking an API key (R13/U10).

What this does instead:
- Raw logs stay in `logs/` (ignored), unchanged, on both boxes.
- Each box names itself once, per machine, in `.env`: `AIT_LOG_OWNER=prithvi`
  (or `ahmed`). Unset = publishing is off; nothing else changes.
- After the close the supervisor exports one ET trading day of the bot's own
  structured JSON events (all levels — IV ranks and gate outcomes are debug),
  secret-scrubbed, gzipped (~0.1-0.3 MB/day), to
  `<owner>_logs/YYYY-MM-DD.jsonl.gz` on a SEPARATE orphan branch,
  `shared-logs`, through a SEPARATE git worktree. The code branch, and the
  live tree the bot runs from, are never committed to, checked out or
  modified — so the deploy path (`merge --ff-only`) cannot be broken by it.
- `scripts/read_shared_logs.py --owner ahmed --date 2026-09-28` reads it back.

Everything here fails closed and quiet: a publishing problem is logged and
retried the next day; it can never touch trading.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

SHARED_BRANCH = "shared-logs"
OWNER_ENV = "AIT_LOG_OWNER"
OWNER_RE = re.compile(r"^[a-z][a-z0-9_-]{1,30}$")
GIT_TIMEOUT_S = 120

# Loggers whose lines are never exported: the stdlib HTTP stack is the R13/U10
# vector (urllib3 DEBUG printed request URLs carrying an API key).
DROP_LOGGER_PREFIXES = ("urllib3", "requests", "httpx", "httpcore", "yfinance", "peewee")

_SECRET_KV = re.compile(
    r"""(?ix)
    (api[_-]?key|apikey|access[_-]?token|token|secret|password|passwd|authorization)
    (\\?["']?\s*[:=]\s*\\?["']?|\s+)
    ([^"'\\&\s,}]{4,})
    """)
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s\"',}]+")
_ACCOUNT_ID = re.compile(r"\b(?:DU[A-Z]?\d{5,}|U\d{7,})\b")


def resolve_owner(environ: dict | None = None) -> str | None:
    """This box's owner name from AIT_LOG_OWNER, or None (publishing off).
    A malformed name is treated as unset, never guessed or normalised."""
    env = os.environ if environ is None else environ
    raw = str(env.get(OWNER_ENV, "") or "").strip()
    return raw if OWNER_RE.match(raw) else None


def scrub(text: str) -> str:
    """Redact key/token/password-shaped values and IBKR account ids."""
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _SECRET_KV.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    return _ACCOUNT_ID.sub("[ACCOUNT]", text)


def _log_files(logs_dir: Path) -> list[Path]:
    """ait.log rotations oldest first: ait.log.N ... ait.log.1, ait.log."""
    rotated = []
    for p in logs_dir.glob("ait.log.*"):
        suffix = p.name.rsplit(".", 1)[-1]
        if suffix.isdigit():
            rotated.append((int(suffix), p))
    ordered = [p for _, p in sorted(rotated, reverse=True)]
    live = logs_dir / "ait.log"
    if live.exists():
        ordered.append(live)
    return ordered


def _event_date_et(obj: dict) -> date | None:
    ts = obj.get("timestamp")
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:        # a naive stamp is host-local; this box is ET
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(ET).date()


def select_day_events(logs_dir: Path, day: date) -> list[str]:
    """The bot's own structured events whose timestamp falls on `day` in ET,
    scrubbed, de-duplicated, in time order. Non-JSON lines (the IBKR
    `Error 10091`/`Warning 10167` prints, tracebacks) are skipped."""
    out: dict[str, str] = {}
    for path in _log_files(logs_dir):
        try:
            with open(path, "rb") as f:
                for raw in f:
                    if not raw.startswith(b"{"):
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if str(obj.get("logger", "")).startswith(DROP_LOGGER_PREFIXES):
                        continue
                    if _event_date_et(obj) != day:
                        continue
                    out.setdefault(line, scrub(line))
        except OSError:
            continue
    return sorted(out.values(), key=lambda s: json.loads(s).get("timestamp", ""))


def export_day(logs_dir: Path, day: date, dest: Path, owner: str) -> dict:
    """Write one day to `dest` (gzip JSONL; first line is a `_meta` header
    recording coverage, so a gap from log rotation or downtime is visible
    rather than silent). Returns the header."""
    events = select_day_events(logs_dir, day)
    first = json.loads(events[0])["timestamp"] if events else None
    last = json.loads(events[-1])["timestamp"] if events else None
    meta = {"_meta": {"owner": owner, "date": day.isoformat(), "events": len(events),
                      "first_event": first, "last_event": last,
                      "exported_at": datetime.now(ET).isoformat(timespec="seconds"),
                      "source": "logs/ait.log* (JSON lines only, scrubbed)"}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for line in events:
            f.write(line + "\n")
    return meta["_meta"]


def read_day(path_or_bytes) -> tuple[dict, list[dict]]:
    """(meta, events) from an exported file path or its raw gzip bytes."""
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) \
        else Path(path_or_bytes).read_bytes()
    lines = gzip.decompress(data).decode("utf-8").splitlines()
    meta = json.loads(lines[0]).get("_meta", {}) if lines else {}
    return meta, [json.loads(x) for x in lines[1:] if x.strip()]


def summarize(events: list[dict]) -> dict:
    """The questions that come up: did it scan, what ranks did it see, what
    stopped entries, what did it place, what went wrong."""
    counts: dict[str, int] = {}
    for e in events:
        counts[e.get("event", "?")] = counts.get(e.get("event", "?"), 0) + 1
    ranks: dict[str, float] = {}
    for e in events:
        if e.get("event") == "iv_rank_true_percentile" and e.get("symbol"):
            ranks[e["symbol"]] = e.get("rank")
    return {
        "scans": counts.get("scan_symbol_timing", 0),
        "iv_ranks_last": ranks,
        "rejections": {k: v for k, v in sorted(counts.items())
                       if k.startswith(("condor_rejected_", "entry_blocked", "economic_event_skip"))
                       or k.endswith("_entry_refused")},
        "orders": {k: counts.get(k, 0) for k in
                   ("trade_validated", "order_placed", "exit_order_registered",
                    "position_exit_triggered")},
        "warnings_errors": sorted(
            {e.get("event") for e in events if e.get("level") in ("warning", "error", "critical")}),
        "first_event": events[0].get("timestamp") if events else None,
        "last_event": events[-1].get("timestamp") if events else None,
    }


# ---------------------------------------------------------------- fetching

_OWNER_DAY_RE = re.compile(
    r"^([a-z][a-z0-9_-]{1,30})_logs/(\d{4}-\d{2}-\d{2})\.jsonl\.gz$")


@dataclass
class FetchResult:
    """Outcome of refreshing `origin/shared-logs` in the live checkout."""

    ok: bool
    advanced: bool = False
    head: str | None = None
    new_commits: int = 0
    latest_by_owner: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def _remote_head(repo: Path) -> str | None:
    r = _git(["rev-parse", "--verify", "--quiet",
              f"refs/remotes/origin/{SHARED_BRANCH}"], repo)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def fetch_shared_branch(repo: Path) -> FetchResult:
    """Refresh `origin/shared-logs` so the OTHER box's published days are readable locally.

    A FETCH ONLY: it updates the remote-tracking ref (and FETCH_HEAD) and nothing else —
    no checkout, no merge, no change to HEAD, the index, the working tree or any local
    branch (the same operation scripts/read_shared_logs.py runs on every invocation, and
    proven by test not to touch the live tree). It does not need AIT_LOG_OWNER: reading a
    peer's logs does not require publishing your own.

    `latest_by_owner` is the newest published day per owner, so a box that has silently
    stopped publishing is visible in the log without anyone looking (2026-10-01 was the
    last day on the branch for one box when this was written).

    GIT_TERMINAL_PROMPT=0: a headless supervisor must never block on a credential prompt.
    Never raises.
    """
    try:
        before = _remote_head(repo)
        r = subprocess.run(
            ["git", "fetch", "--quiet", "origin", SHARED_BRANCH], cwd=str(repo),
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "").strip()[-300:]
            return FetchResult(ok=False, head=(before or "")[:12] or None,
                               error=msg or f"git fetch exited {r.returncode}")
        after = _remote_head(repo)
        if not after:
            return FetchResult(ok=False, error=f"origin/{SHARED_BRANCH} not present after fetch")
        span = f"{before}..{after}" if before else after
        c = _git(["rev-list", "--count", span], repo)
        new = int(c.stdout.strip() or 0) if c.returncode == 0 and before != after else 0
        latest: dict[str, str] = {}
        t = _git(["ls-tree", "-r", "--name-only", after], repo)
        for line in (t.stdout.splitlines() if t.returncode == 0 else []):
            m = _OWNER_DAY_RE.match(line.strip())
            if m and m.group(2) > latest.get(m.group(1), ""):
                latest[m.group(1)] = m.group(2)
        return FetchResult(ok=True, advanced=(before != after), head=after[:12],
                           new_commits=new, latest_by_owner=latest)
    except Exception as e:  # noqa: BLE001 — a fetch must never hurt the supervisor
        return FetchResult(ok=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- publishing

@dataclass
class PublishResult:
    ok: bool
    published: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    error: str | None = None


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=GIT_TIMEOUT_S)


def _remote_has_branch(repo: Path) -> bool:
    r = _git(["ls-remote", "--heads", "origin", SHARED_BRANCH], repo)
    return r.returncode == 0 and SHARED_BRANCH in r.stdout


def worktree_dir_for(repo: Path) -> Path:
    return repo.parent / f"{repo.name}_shared_logs"


def _ensure_worktree(repo: Path, wt: Path) -> None:
    """A worktree of the orphan `shared-logs` branch, beside the repo. Never
    checks anything out in `repo` itself."""
    if (wt / ".git").exists():
        return
    if wt.exists():
        shutil.rmtree(wt)
    _git(["worktree", "prune"], repo)
    if _remote_has_branch(repo):
        _git(["fetch", "origin", SHARED_BRANCH], repo)
        r = _git(["worktree", "add", "-B", SHARED_BRANCH, str(wt),
                  f"origin/{SHARED_BRANCH}"], repo)
    else:
        r = _git(["worktree", "add", "--orphan", "-b", SHARED_BRANCH, str(wt)], repo)
        if r.returncode == 0:
            (wt / "README.md").write_text(
                "# shared-logs\n\nEach trading box's own bot events, one gzip JSONL per ET "
                "trading day: `<owner>_logs/YYYY-MM-DD.jsonl.gz`.\nWritten by "
                "`src/ait/orchestration/shared_logs.py`; read with "
                "`scripts/read_shared_logs.py`. Orphan branch — never merge into code.\n",
                encoding="utf-8")
            _git(["add", "README.md"], wt)
            _git(["commit", "-m", "shared-logs: start the branch"], wt)
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed: {r.stderr.strip()[:300]}")


def publish(repo: Path, owner: str, days: list[date], logs_dir: Path | None = None,
            worktree: Path | None = None, retries: int = 3) -> PublishResult:
    """Export each of `days` that has events and push them to `shared-logs`.
    Days already published are re-exported only if the local logs now hold
    more events than the published file (a same-day re-run after more
    activity). Never raises."""
    logs_dir = logs_dir or (repo / "logs")
    wt = worktree or worktree_dir_for(repo)
    res = PublishResult(ok=False)
    try:
        if not OWNER_RE.match(owner or ""):
            res.error = f"invalid owner {owner!r}"
            return res
        _ensure_worktree(repo, wt)
        if _remote_has_branch(repo):
            _git(["pull", "--rebase", "origin", SHARED_BRANCH], wt)
        folder = wt / f"{owner}_logs"
        for d in days:
            dest = folder / f"{d.isoformat()}.jsonl.gz"
            n_new = len(select_day_events(logs_dir, d))
            if n_new == 0:
                res.skipped.append(f"{d}: no events")
                continue
            if dest.exists():
                try:
                    n_old = read_day(dest)[0].get("events", 0)
                except Exception:  # noqa: BLE001 — unreadable file: overwrite it
                    n_old = -1
                if n_old >= n_new:
                    res.skipped.append(f"{d}: already published ({n_old} events)")
                    continue
            export_day(logs_dir, d, dest, owner)
            res.published.append(d.isoformat())
        if not res.published:
            res.ok = True
            return res
        _git(["add", f"{owner}_logs"], wt)
        c = _git(["commit", "-m", f"logs({owner}): {', '.join(res.published)}"], wt)
        if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr):
            res.error = f"commit failed: {(c.stderr or c.stdout).strip()[:300]}"
            return res
        for _ in range(max(1, retries)):
            p = _git(["push", "origin", f"HEAD:{SHARED_BRANCH}"], wt)
            if p.returncode == 0:
                res.ok = True
                return res
            # Another box pushed first: its commits only touch its own folder,
            # so a rebase is conflict-free.
            _git(["pull", "--rebase", "origin", SHARED_BRANCH], wt)
        res.error = f"push failed: {p.stderr.strip()[:300]}"
        return res
    except Exception as e:  # noqa: BLE001 — publishing must never raise
        res.error = f"{type(e).__name__}: {e}"
        return res


def recent_trading_days(today: date, n: int = 5) -> list[date]:
    """The last `n` weekdays up to and including `today` (catch-up for days
    the box was off at publish time; holidays simply have no events)."""
    out, d = [], today
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)
