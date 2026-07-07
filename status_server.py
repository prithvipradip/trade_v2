#!/usr/bin/env python
"""Live AIT bot dashboard — a browser page that streams status in real time.

Serves a single auto-refreshing page at http://localhost:8503 that polls the
bot's state every 3s and updates in place. Reads the same DB/log snapshot as
status.py (no IBKR connection), so it's safe to run alongside the bot.

Run:  python status_server.py     (or launched by the supervisor on boot)
"""

from __future__ import annotations

import json
from flask import Flask, Response

from status import get_status

app = Flask(__name__)

PAGE = """<!doctype html><html><head>
<meta charset="utf-8"><title>AIT Bot — Live</title>
<style>
 body{background:#0d1117;color:#c9d1d9;font:14px/1.5 ui-monospace,Consolas,monospace;margin:0;padding:24px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#8b949e;font-size:12px;margin-bottom:18px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
 .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#8b949e;margin:0 0 10px}
 .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d}
 .row:last-child{border:0} .k{color:#8b949e} .v{font-weight:600}
 .ok{color:#3fb950}.warn{color:#d29922}.bad{color:#f85149}.big{font-size:26px;font-weight:700}
 table{width:100%;border-collapse:collapse;font-size:13px} td{padding:4px 6px;border-bottom:1px solid #21262d}
 .pulse{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
</style></head><body>
<h1><span id="dot" class="pulse"></span>AIT Bot — Live</h1>
<div class="sub" id="asof">connecting…</div>
<div class="grid">
  <div class="card"><h2>Health</h2><div id="health"></div></div>
  <div class="card"><h2>Today</h2><div id="today"></div></div>
  <div class="card"><h2>Realized P&amp;L (real exits)</h2><div id="pnl"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Open Positions</h2><div id="pos"></div></div>
</div>
<script>
const E=(id)=>document.getElementById(id);
function row(k,v,cls){return `<div class="row"><span class="k">${k}</span><span class="v ${cls||''}">${v}</span></div>`}
async function tick(){
 try{
  const s=await (await fetch('/api/status')).json();
  E('asof').textContent='as of '+s.asof+'  ·  auto-refresh 3s';
  E('dot').style.background=s.running?'#3fb950':'#f85149';
  let h='';
  h+=row('Running', s.running?'YES':'NO', s.running?'ok':'bad');
  h+=row('Processes', s.procs);
  h+=row('Last heartbeat', s.last_heartbeat);
  h+=row('Last connect (UTC)', s.last_connect);
  h+=row('Native crashes', s.crashes, s.crashes>0?'warn':'ok');
  h+=row('Keeper relaunches', s.keeper_relaunches);
  h+=row('Read-only', s.readonly?'YES — orders rejected!':'clear', s.readonly?'bad':'ok');
  h+=row('Last activity', s.last_activity);
  E('health').innerHTML=h;
  let t=''; for(const[k,v]of Object.entries(s.today)) t+=row(k, v, k==='trade_executed'&&v>0?'ok':'');
  E('today').innerHTML=t;
  E('pnl').innerHTML=
    `<div class="big ${s.pnl_today>=0?'ok':'bad'}">$${s.pnl_today.toFixed(2)}</div>`+
    row('today closed', s.pnl_today_n)+
    row('lifetime', '$'+s.pnl_life.toFixed(2), s.pnl_life>=0?'ok':'bad')+
    row('lifetime closed', s.pnl_life_n)+
    row('open unrealized', (s.unrealized_total>=0?'+$':'-$')+Math.abs(s.unrealized_total||0).toFixed(2),
        (s.unrealized_total||0)>=0?'ok':'bad');
  if(s.open_positions.length){
    let r='<table><tr class="k"><td>sym</td><td>strategy</td><td>status</td><td>entry</td><td>unrealized</td><td>%</td><td>since</td></tr>';
    for(const p of s.open_positions){
      const u=p.unrealized||0, cls=u>0?'ok':(u<0?'bad':'k');
      r+=`<tr><td><b>${p.symbol}</b></td><td>${p.strategy}</td><td>${p.status}</td>`+
         `<td>${p.entry}</td><td class="${cls}"><b>${u>=0?'+':''}$${u.toFixed(2)}</b></td>`+
         `<td class="${cls}">${(p.pnl_pct||0)>=0?'+':''}${p.pnl_pct||0}%</td><td>${p.since}</td></tr>`;
    }
    E('pos').innerHTML=r+'</table>';
  } else E('pos').innerHTML='<span class="k">none — flat</span>';
 }catch(e){ E('asof').textContent='dashboard offline / bot stopped'; E('dot').style.background='#f85149'; }
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/status")
def api_status():
    return Response(json.dumps(get_status()), mimetype="application/json")


if __name__ == "__main__":
    print("AIT live dashboard -> http://localhost:8503")
    app.run(host="127.0.0.1", port=8503, threaded=True)
