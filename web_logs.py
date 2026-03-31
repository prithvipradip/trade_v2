#!/usr/bin/env python
"""Web-based log viewer for AIT v2.

Open http://localhost:8502 in your browser to see live trading logs.
Auto-refreshes every 5 seconds.
"""

import json
import time
from pathlib import Path
from datetime import datetime

from flask import Flask, Response, render_template_string

app = Flask(__name__)

LOG_FILE = Path(__file__).parent / "logs" / "bot_stdout.log"
ORCH_LOG = Path(__file__).parent / "logs" / "orchestrator.log"

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
        <div class="stat"><div class="stat-label">Last Signal</div><div class="stat-value" id="last-signal">-</div></div>
    </div>

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
                    document.getElementById('last-signal').textContent = data.last_signal;
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

def parse_log_line(line):
    """Parse a structured log line into displayable HTML."""
    line = line.strip()
    if not line:
        return None

    # Skip noise
    skip = ['UserWarning', 'warnings.warn', 'HTTP Request', 'Connection pool',
            'Loading weights', 'UNEXPECTED', 'position_ids']
    if any(s in line for s in skip):
        return None

    # Try JSON structured log
    try:
        data = json.loads(line)
        event = data.get('event', '')
        component = data.get('component', '')
        level = data.get('level', 'info')
        ts = data.get('timestamp', '')[:19].replace('T', ' ')
        symbol = data.get('symbol', '')

        # Classify
        if 'order_placed' in event or 'trade_executed' in event or 'filled' in event:
            css = 'trade'
            category = 'trade'
        elif 'error' in level or 'critical' in level:
            css = 'error'
            category = 'error'
        elif 'warning' in level or 'reject' in event:
            css = 'warning'
            category = 'error'
        elif 'prediction' in event or 'ml_prediction' in event:
            css = 'prediction'
            category = 'prediction'
        elif 'signal' in event:
            css = 'signal'
            category = 'signal'
        else:
            css = 'info'
            category = 'info'

        # Format
        sym_html = f'<span class="symbol">{symbol}</span> ' if symbol else ''
        extras = {k: v for k, v in data.items()
                  if k not in ('event', 'component', 'level', 'logger', 'timestamp', 'symbol')}
        extras_str = ' | '.join(f'{k}={v}' for k, v in extras.items()) if extras else ''

        html = (f'<div class="{css}" data-category="{category}">'
                f'<span class="timestamp">{ts}</span> '
                f'{sym_html}'
                f'<strong>{event}</strong> '
                f'<span class="info">[{component}]</span> '
                f'{extras_str}</div>')
        return html, category, event, symbol
    except (json.JSONDecodeError, Exception):
        pass

    # Plain text lines (orchestrator, errors)
    if 'Error' in line or 'error' in line:
        return f'<div class="error">{line}</div>', 'error', '', ''
    if 'Warning' in line or 'Canceled' in line:
        return f'<div class="warning">{line}</div>', 'error', '', ''

    return f'<div class="info">{line}</div>', 'info', '', ''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/logs')
def api_logs():
    from flask import request
    log_filter = request.args.get('filter', 'all')

    lines = []
    trades_today = 0
    last_signal = '-'
    status = 'Unknown'
    today = datetime.now().strftime('%Y-%m-%d')

    # Read last 200 lines of bot log
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', errors='replace') as f:
            all_lines = f.readlines()
            recent = all_lines[-200:]
    else:
        recent = []

    html_lines = []
    for line in recent:
        result = parse_log_line(line)
        if result is None:
            continue
        html, category, event, symbol = result

        # Count today's trades
        if today in line and ('trade_executed' in event or 'order_placed' in event):
            trades_today += 1

        # Track last signal
        if 'ml_prediction' in event and symbol:
            last_signal = symbol

        # Filter
        if log_filter == 'all' or category == log_filter:
            html_lines.append(html)

    # Check orchestrator for status
    if ORCH_LOG.exists():
        with open(ORCH_LOG, 'r', errors='replace') as f:
            orch_lines = f.readlines()[-5:]
        for l in orch_lines:
            if 'bot_healthy' in l:
                status = 'Healthy'
            elif 'bot_down' in l:
                status = 'DOWN'
            elif 'bot_started' in l:
                status = 'Starting'

    return {
        'html': '\n'.join(html_lines[-100:]),
        'status': status,
        'trades_today': str(trades_today),
        'last_signal': last_signal,
    }


if __name__ == '__main__':
    print("=" * 50)
    print("  AIT v2 Web Log Viewer")
    print("  Open http://localhost:8502")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8502, debug=False)
