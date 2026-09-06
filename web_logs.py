#!/usr/bin/env python
"""Web-based log viewer for AIT v2.

Open http://localhost:8502 in your browser to see live trading logs.
Auto-refreshes every 5 seconds.

W6 log-contracts-1 / -4 / -6.  This viewer used to:
  * read the ENTIRE log with ``f.readlines()`` and keep the last 200 lines —
    on the measured production file that is 76 MB re-read every 5 s per open
    browser tab, and 200 lines covers well under a minute of RTH, so a trade
    executed 20 minutes earlier was invisible and "Trades Today" showed 0;
  * bucket "today" by substring-matching a LOCAL date against UTC timestamps;
  * count trade_executed AND order_placed into one "trades" tile (order_placed
    fires per order, entries and exits alike);
  * parse only raw JSON, so an interactive-console start (ANSI ConsoleRenderer
    lines) made every event unparseable;
  * report "Healthy" from the last orchestrator.log line no matter how old.

All five are fixed here; the shared parsing/tailing/counting authority is
ait.monitoring.ops_health, so status.py and the dashboard get the same fixes.
"""

import html as _html
import sys
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template_string

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ait.monitoring.ops_health import (  # noqa: E402
    DEFAULT_TAIL_BYTES,
    IncrementalEventCounter,
    parse_event_line,
    parse_timestamp,
    strip_ansi,
    tail_lines,
)

app = Flask(__name__)

LOG_FILE = Path(__file__).parent / "logs" / "bot_stdout.log"
ORCH_LOG = Path(__file__).parent / "logs" / "orchestrator.log"

#: Bytes of the tail rendered in the scrollback pane. 512 KB is ~2-4k lines of
#: production log — orders of magnitude more history than the old 200 lines,
#: and a bounded seek instead of an O(file) read.
DISPLAY_TAIL_BYTES = DEFAULT_TAIL_BYTES

#: Events the day counters track. Separated on purpose: order_placed fires for
#: every order (entry legs AND exits), so folding it into "trades" inflated the
#: count; trade_executed is one per executed trade (executor.py:443).
COUNTED_EVENTS = (
    "trade_executed", "order_placed", "ml_prediction",
    "signals_generated", "trade_rejected",
)

#: The supervisor logs bot_healthy every ~2 min (master.py:486). Past this the
#: last line is history, not a status — reporting "Healthy" off a three-day-old
#: line is the same false all-clear this whole work item is about.
ORCH_STATUS_MAX_AGE_S = 600.0

# One counter per process: seeded once from today's rotated backups plus a
# byte-capped tail, then only the newly appended bytes are ever read again.
# min_interval_s throttles the 5 s poll of every open tab down to one stat()
# per 5 s regardless of tab count.
_COUNTER = IncrementalEventCounter(
    LOG_FILE, COUNTED_EVENTS, min_interval_s=5.0,
)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AIT v2 - Live Trading Logs</title>
    <style>
        body { background: #0d1117; color: #c9d1d9; font-family: 'Consolas', monospace; margin: 0; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 5px; }
        .subtitle { color: #8b949e; margin-bottom: 20px; }
        .controls { margin-bottom: 15px; }
        .controls button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 16px; cursor: pointer; border-radius: 6px; margin-right: 8px; }
        .controls button:hover { background: #30363d; }
        .controls button.active { background: #1f6feb; border-color: #1f6feb; }
        #logs { height: 70vh; overflow-y: auto; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 13px; line-height: 1.6; }
        .trade { color: #3fb950; font-weight: bold; }
        .error { color: #f85149; }
        .warning { color: #d29922; }
        .prediction { color: #a5d6ff; }
        .signal { color: #d2a8ff; }
        .info { color: #8b949e; }
        .timestamp { color: #6e7681; }
        .symbol { color: #ffa657; font-weight: bold; }
        .stats { display: flex; gap: 20px; margin-bottom: 15px; }
        .stat { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 20px; }
        .stat-label { color: #8b949e; font-size: 12px; }
        .stat-value { color: #58a6ff; font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>AIT v2 Trading Bot</h1>
    <div class="subtitle">Live Log Viewer | Auto-refreshes every 5s</div>

    <div class="stats" id="stats">
        <div class="stat"><div class="stat-label">Status</div><div class="stat-value" id="status">Loading...</div></div>
        <div class="stat"><div class="stat-label">Trades Today</div><div class="stat-value" id="trades">-</div></div>
        <div class="stat"><div class="stat-label">Orders Today</div><div class="stat-value" id="orders">-</div></div>
        <div class="stat"><div class="stat-label">ML Preds Today</div><div class="stat-value" id="preds">-</div></div>
        <div class="stat"><div class="stat-label">Last Signal</div><div class="stat-value" id="last-signal">-</div></div>
    </div>
    <div class="subtitle" id="counter-note"></div>

    <div class="controls">
        <button class="active" onclick="setFilter('all')">All</button>
        <button onclick="setFilter('trade')">Trades</button>
        <button onclick="setFilter('prediction')">ML Predictions</button>
        <button onclick="setFilter('signal')">Signals</button>
        <button onclick="setFilter('error')">Errors</button>
    </div>

    <div id="logs"></div>

    <script>
        let filter = 'all';
        let autoScroll = true;

        function setFilter(f) {
            filter = f;
            document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            fetchLogs();
        }

        function fetchLogs() {
            fetch('/api/logs?filter=' + filter)
                .then(r => r.json())
                .then(data => {
                    document.getElementById('logs').innerHTML = data.html;
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('trades').textContent = data.trades_today;
                    document.getElementById('orders').textContent = data.orders_today;
                    document.getElementById('preds').textContent = data.predictions_today;
                    document.getElementById('last-signal').textContent = data.last_signal;
                    document.getElementById('counter-note').textContent = data.counter_note;
                    if (autoScroll) {
                        let el = document.getElementById('logs');
                        el.scrollTop = el.scrollHeight;
                    }
                });
        }

        setInterval(fetchLogs, 5000);
        fetchLogs();
    </script>
</body>
</html>
"""

_SKIP_NOISE = ('UserWarning', 'warnings.warn', 'HTTP Request', 'Connection pool',
               'Loading weights', 'UNEXPECTED', 'position_ids')

_HIDDEN_FIELDS = ('event', 'component', 'level', 'logger', 'timestamp', 'symbol')


def classify(event, level):
    """(css class, filter category) for one event/level pair."""
    event = event or ''
    level = level or 'info'
    if 'order_placed' in event or 'trade_executed' in event or 'filled' in event:
        return 'trade', 'trade'
    if 'error' in level or 'critical' in level:
        return 'error', 'error'
    if 'warning' in level or 'reject' in event:
        return 'warning', 'error'
    if 'prediction' in event:
        return 'prediction', 'prediction'
    if 'signal' in event:
        return 'signal', 'signal'
    return 'info', 'info'


def parse_log_line(line):
    """Parse one log line into ``(html, category, event, symbol)`` or None.

    log-contracts-6: delegates to ops_health.parse_event_line, which reads the
    JSON rendering AND the ConsoleRenderer rendering (with or without ANSI
    escapes).  Before this, a bot started from an interactive console wrote
    ANSI lines that no consumer could parse — silently, with no error anywhere.
    """
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    if any(s in line for s in _SKIP_NOISE):
        return None

    rec = parse_event_line(line)
    if rec is not None:
        event = str(rec.get('event', ''))
        component = str(rec.get('component', ''))
        level = str(rec.get('level', 'info'))
        symbol = str(rec.get('symbol', '') or '')
        dt = parse_timestamp(rec.get('timestamp'))
        ts = dt.strftime('%Y-%m-%d %H:%M:%S') if dt else ''

        css, category = classify(event, level)
        sym_html = (f'<span class="symbol">{_html.escape(symbol)}</span> '
                    if symbol else '')
        extras = {k: v for k, v in rec.items()
                  if k not in _HIDDEN_FIELDS and not k.startswith('_')}
        extras_str = ' | '.join(
            f'{_html.escape(str(k))}={_html.escape(str(v))}'
            for k, v in extras.items())
        html = (f'<div class="{css}" data-category="{category}">'
                f'<span class="timestamp">{ts}</span> '
                f'{sym_html}'
                f'<strong>{_html.escape(event)}</strong> '
                f'<span class="info">[{_html.escape(component)}]</span> '
                f'{extras_str}</div>')
        return html, category, event, symbol

    # Plain text lines (ib_insync chatter, tracebacks). ANSI stripped so a
    # console-rendered session is still readable rather than full of escapes.
    plain = _html.escape(strip_ansi(line))
    if 'Error' in line or 'error' in line:
        return f'<div class="error">{plain}</div>', 'error', '', ''
    if 'Warning' in line or 'Canceled' in line:
        return f'<div class="warning">{plain}</div>', 'error', '', ''
    return f'<div class="info">{plain}</div>', 'info', '', ''


def orchestrator_status(path=ORCH_LOG, *, now=None, max_age_s=ORCH_STATUS_MAX_AGE_S):
    """Supervisor status from orchestrator.log, WITH a staleness check.

    The old code took the last bot_healthy/bot_down/bot_started line it saw and
    reported it forever — a master that died three days ago still read
    "Healthy".  A supervisor line older than ``max_age_s`` is history.
    """
    now = now or datetime.now()
    lines = tail_lines(path, 64 * 1024)
    status = 'Unknown'
    stamp = None
    for ln in lines[-40:]:
        plain = strip_ansi(ln)
        if 'bot_healthy' in plain:
            status = 'Healthy'
        elif 'bot_down' in plain:
            status = 'DOWN'
        elif 'bot_started' in plain:
            status = 'Starting'
        else:
            continue
        # master.py:56 writes "[2026-08-31 13:46:51] DEBUG orchestrator.<event>"
        head = plain.split(']', 1)[0].lstrip('[') if plain.startswith('[') else ''
        stamp = parse_timestamp(head) or stamp
    if status == 'Unknown':
        return 'No supervisor log', None
    if stamp is None:
        return f'{status} (unstamped)', None
    age = (now.astimezone() - stamp).total_seconds()
    if age > max_age_s:
        return f'{status} @ {stamp:%H:%M} (STALE {age / 60:.0f}m)', age
    return status, age


@app.route('/')
def index():
    return render_template_string(HTML)


def day_counts(counter=None, *, now=None):
    """Today's event counts + last ML-prediction symbol.

    log-contracts-4: the counters are NO LONGER derived from the display
    window.  They come from an incremental scan of the whole day (rotated
    backups included, log-contracts-1) that reads only newly-appended bytes.
    log-contracts-1/time-authority-1: the day bucket is the LOCAL date of each
    record's UTC timestamp, not a substring match of a local date string
    against a UTC stamp.
    """
    counter = counter if counter is not None else _COUNTER
    snap = counter.refresh(now=now)
    last_pred = snap.last_by_event.get('ml_prediction') or {}
    last_symbol = str(last_pred.get('symbol') or '-')
    note = f"counters: {snap.day} from {', '.join(snap.sources) or 'no log file'}"
    if snap.partial:
        note += " (byte-capped/rotated: counts may under-report)"
    return snap, last_symbol, note


@app.route('/api/logs')
def api_logs():
    from flask import request
    log_filter = request.args.get('filter', 'all')

    # Bounded tail for the DISPLAY pane (was: readlines() over a 76 MB file
    # every 5 s per tab, then the last 200 lines).
    recent = tail_lines(LOG_FILE, DISPLAY_TAIL_BYTES)

    html_lines = []
    for line in recent:
        result = parse_log_line(line)
        if result is None:
            continue
        html, category, _event, _symbol = result
        if log_filter == 'all' or category == log_filter:
            html_lines.append(html)

    snap, last_signal, counter_note = day_counts()
    status, _age = orchestrator_status()

    return {
        'html': '\n'.join(html_lines[-400:]),
        'status': status,
        'trades_today': str(snap.counts.get('trade_executed', 0)),
        'orders_today': str(snap.counts.get('order_placed', 0)),
        'predictions_today': str(snap.counts.get('ml_prediction', 0)),
        'signals_today': str(snap.counts.get('signals_generated', 0)),
        'rejects_today': str(snap.counts.get('trade_rejected', 0)),
        'last_signal': last_signal,
        'counter_note': counter_note,
    }


if __name__ == '__main__':
    print("=" * 50)
    print("  AIT v2 Web Log Viewer")
    print("  Open http://localhost:8502")
    print("=" * 50)
    # localhost ONLY (deep-audit S1): 0.0.0.0 exposed the raw live trade
    # stream (positions, orders, P&L) to the whole LAN with no auth.
    app.run(host='127.0.0.1', port=8502, debug=False)
