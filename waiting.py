# ═══════════════════════════════════════════════════════════════════════════════
#  BAT 3301 – Colazo – Waiting Lines
#  M/M/c queues · Erlang-C model · Poisson arrivals · exponential service
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Waiting Lines – BAT 3301",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #ffffff; color: #0f172a; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f8f9fc; border-radius: 8px;
    padding: 4px; border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-weight: 500; font-size: 14px;
    color: #64748b; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #1e40af !important;
    font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.page-title {
    font-size: 26px; font-weight: 800; letter-spacing: -0.8px;
    color: #0f172a; margin-bottom: 2px;
}
.page-sub { font-size: 13px; color: #94a3b8; margin-bottom: 20px; }
.sec-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    color: #94a3b8; text-transform: uppercase;
    border-bottom: 1px solid #e2e8f0; padding-bottom: 7px; margin-bottom: 14px;
}
.kpi-card {
    background: #f8f9fc; border: 1px solid #e2e8f0;
    border-top: 3px solid; border-radius: 6px; padding: 12px 14px;
}
.kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    color: #94a3b8; text-transform: uppercase;
}
.kpi-val { font-size: 22px; font-weight: 700; margin-top: 5px; letter-spacing: -0.5px; }
.info-box {
    background: #f8f9fc; border: 1px solid #e2e8f0; border-left: 4px solid;
    border-radius: 0 6px 6px 0; padding: 12px 16px;
    font-size: 13px; line-height: 1.75; color: #374151; margin-bottom: 12px;
}
.warn-box {
    background: #fef2f2; border: 1px solid #fca5a5; border-radius: 6px;
    padding: 12px 16px; font-size: 13px; color: #7f1d1d;
}
.metric-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.metric-tbl th {
    background:#f1f5f9; text-align:left; padding:9px 12px;
    font-weight:600; color:#374151; border:1px solid #e2e8f0; font-size:12px;
}
.metric-tbl td { padding:8px 12px; border:1px solid #e2e8f0; color:#374151; }
.metric-tbl tr:nth-child(even) td { background:#f8f9fc; }
.metric-tbl td.hl { font-weight:700; }
.cmp-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.cmp-tbl th { background:#f1f5f9; text-align:center; padding:9px 12px; font-weight:600; border:1px solid #e2e8f0; }
.cmp-tbl td { padding:8px 12px; border:1px solid #e2e8f0; text-align:center; }
.cmp-tbl tr:nth-child(even) td { background:#f8f9fc; }
.cmp-tbl td.better { color:#16a34a; font-weight:700; }
.cmp-tbl td.worse  { color:#dc2626; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
BLUE   = "#2563eb"
GREEN  = "#16a34a"
RED    = "#dc2626"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
TEAL   = "#0891b2"
GRAY   = "#6b7280"

SERVER_COLORS = [BLUE, GREEN, PURPLE, TEAL]
LANE_COLORS   = [BLUE, ORANGE, GREEN, PURPLE]

def hex_rgba(h, a=0.12):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

# ═══════════════════════════════════════════════════════════════════════════════
#  M/M/c MATH
# ═══════════════════════════════════════════════════════════════════════════════
def mmc(lam: float, mu: float, c: int) -> dict | None:
    if lam <= 0 or mu <= 0 or c < 1:
        return None
    a   = lam / mu
    rho = a / c
    if rho >= 1.0:
        return None
    s    = sum(a**n / math.factorial(n) for n in range(c))
    last = a**c / (math.factorial(c) * (1 - rho))
    P0   = 1.0 / (s + last)
    Pw   = last * P0
    Lq   = Pw * rho / (1 - rho)
    Wq   = Lq / lam
    W    = Wq + 1.0 / mu
    L    = lam * W
    Ls   = a
    return dict(lam=lam, mu=mu, c=c, a=a, rho=rho,
                P0=P0, Pw=Pw, Lq=Lq, L=L, Ls=Ls, Wq=Wq, W=W)

def fmt(v, decimals=4, pct=False, time=False):
    if v is None: return "—"
    if pct:  return f"{v:.2%}"
    if time: return f"{v:.4f}"
    return f"{v:.{decimals}f}"

# ═══════════════════════════════════════════════════════════════════════════════
#  ANIMATED DIAGRAM BUILDERS  (HTML + JS canvas)
#
#  Time scale: animation runs at 1 customer/second entering the total system.
#  Service time in animation = (λ/μ) × 1000 ms  = a × 1000 ms.
#  Scale label notes the conversion: 1 animation second = 1/λ time units.
# ═══════════════════════════════════════════════════════════════════════════════

def _server_colors_js(n: int) -> str:
    """Return JS array literal of server colors."""
    cols = [SERVER_COLORS[i % len(SERVER_COLORS)] for i in range(n)]
    return "[" + ",".join(f'"{c}"' for c in cols) + "]"


def anim_mmc_html(lam: float, mu: float, c: int, rho: float) -> str:
    """
    Animated M/M/c diagram: single shared queue feeding c servers.
    One customer enters per animation second (= 1/λ time units).
    """
    n_rows   = max(c, 2)
    H        = n_rows * 110 + 50
    a_val    = lam / mu
    svc_ms   = round(a_val * 1000, 1)   # ms per service in animation
    scale_s  = round(1 / lam, 4)        # real time units per animation second
    sc_cols  = _server_colors_js(c)

    return f"""<!DOCTYPE html><html><head>
<style>
  body {{margin:0;padding:0;background:#fff;font-family:'Inter',sans-serif;}}
  canvas {{display:block;margin:0 auto;}}
  #note {{font-size:10px;color:#94a3b8;text-align:center;padding:3px 0 4px;}}
</style></head><body>
<canvas id="cv" width="760" height="{H}"></canvas>
<div id="note">Animation scale: 1 second ≈ {scale_s} time units &nbsp;|&nbsp;
λ = {lam:.2f} &nbsp;·&nbsp; μ = {mu:.2f} &nbsp;·&nbsp; c = {c} servers &nbsp;·&nbsp; ρ = {rho:.1%}</div>
<script>
const W = 760, H = {H};
const LAM = {lam}, MU = {mu}, C = {c}, RHO = {rho};
const SPAWN_MS = 1000;
const SVC_MS   = {svc_ms};
const COLS = {sc_cols};

// Layout
const N_ROWS = Math.max(C, 2);
const ROW_H  = (H - 20) / N_ROWS;
const CY     = H / 2;
const serverYs = Array.from({{length: C}}, (_, i) => 10 + ROW_H * i + ROW_H / 2);
const Q_END = 330, S_X0 = 390, S_X1 = 570;

let servers = new Array(C).fill(null);
let queue   = [];
let custs   = [];
let nextId  = 0, lastSpawn = -99999;

function Customer() {{
  this.id    = nextId++;
  this.x     = 5;
  this.y     = CY;
  this.tx    = Q_END;
  this.ty    = CY;
  this.state = 'arriving';
  this.sIdx  = -1;
  this.sEnd  = 0;
  this.col   = '#f97316';
}}

function dispatch(now) {{
  for (let s = 0; s < C; s++) {{
    if (!servers[s] && queue.length > 0) {{
      let cu = queue.shift();
      servers[s] = cu;
      cu.sIdx = s;
      cu.tx = (S_X0 + S_X1) / 2;
      cu.ty = serverYs[s];
      cu.state = 'to_server';
    }}
  }}
}}

function updateCust(cu, dt, now) {{
  switch (cu.state) {{
    case 'arriving':
      cu.x += 180 * dt;
      if (cu.x >= Q_END) {{
        cu.x = Q_END; cu.state = 'queued';
        queue.push(cu); dispatch(now);
      }}
      break;
    case 'queued': {{
      let qi = queue.indexOf(cu);
      let tx = Q_END - 22 * (qi + 1);
      cu.x += (tx - cu.x) * 10 * dt;
      break;
    }}
    case 'to_server': {{
      let dx = cu.tx - cu.x, dy = cu.ty - cu.y;
      let d  = Math.sqrt(dx*dx + dy*dy);
      if (d < 3) {{
        cu.x = cu.tx; cu.y = cu.ty;
        cu.state = 'serving';
        cu.sEnd  = now + SVC_MS;
        cu.col   = '#16a34a';
      }} else {{
        cu.x += dx/d * 250 * dt;
        cu.y += dy/d * 250 * dt;
      }}
      break;
    }}
    case 'serving':
      if (now >= cu.sEnd) {{
        cu.state = 'leaving'; cu.col = '#94a3b8';
        servers[cu.sIdx] = null;
        dispatch(now);
      }}
      break;
    case 'leaving':
      cu.x += 220 * dt;
      if (cu.x > W + 20) cu.state = 'done';
      break;
  }}
}}

function drawStatic(ctx) {{
  // Arrival line (center)
  ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1.8;
  ctx.beginPath(); ctx.moveTo(10, CY); ctx.lineTo(Q_END, CY); ctx.stroke();
  // Arrowhead
  ctx.fillStyle = '#94a3b8';
  ctx.beginPath(); ctx.moveTo(Q_END, CY);
  ctx.lineTo(Q_END - 10, CY - 5); ctx.lineTo(Q_END - 10, CY + 5); ctx.fill();
  // λ label
  ctx.fillStyle = '#374151'; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'left';
  ctx.fillText('λ = ' + LAM.toFixed(2), 12, CY - 13);
  // Queue boundary
  ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(Q_END, 6); ctx.lineTo(Q_END, H - 6); ctx.stroke();
  // Queue label
  ctx.fillStyle = '#94a3b8'; ctx.font = '10px Inter'; ctx.textAlign = 'center';
  ctx.fillText('Queue', Q_END / 2 + 10, 14);

  for (let i = 0; i < C; i++) {{
    let sy  = serverYs[i];
    let col = COLS[i];
    let r   = parseInt(col.slice(1,3),16), g = parseInt(col.slice(3,5),16), b = parseInt(col.slice(5,7),16);
    // Fan line
    ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(Q_END, CY); ctx.lineTo(S_X0, sy); ctx.stroke();
    ctx.setLineDash([]);
    // Departure line
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(S_X1, sy); ctx.lineTo(W - 6, sy); ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.beginPath(); ctx.moveTo(W - 6, sy);
    ctx.lineTo(W - 16, sy - 5); ctx.lineTo(W - 16, sy + 5); ctx.fill();
    // Server box
    ctx.strokeStyle = col; ctx.lineWidth = 2.2;
    ctx.fillStyle   = 'rgba(' + r + ',' + g + ',' + b + ',0.07)';
    ctx.fillRect(S_X0, sy - 28, S_X1 - S_X0, 56);
    ctx.strokeRect(S_X0, sy - 28, S_X1 - S_X0, 56);
    ctx.fillStyle = col; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Server ' + (i + 1), (S_X0 + S_X1) / 2, sy - 7);
    ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter';
    ctx.fillText('μ = ' + MU.toFixed(2) + '   ρ = ' + (RHO * 100).toFixed(1) + '%',
                 (S_X0 + S_X1) / 2, sy + 11);
  }}
}}

const cv  = document.getElementById('cv');
const ctx = cv.getContext('2d');
let prev  = null;

function loop(ts) {{
  if (!prev) prev = ts;
  let dt = (ts - prev) / 1000; prev = ts;
  dt = Math.min(dt, 0.05);

  if (ts - lastSpawn >= SPAWN_MS) {{
    custs.push(new Customer()); lastSpawn = ts;
  }}
  custs.forEach(cu => updateCust(cu, dt, ts));
  custs = custs.filter(cu => cu.state !== 'done');

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  drawStatic(ctx);
  custs.forEach(cu => {{
    ctx.beginPath(); ctx.arc(cu.x, cu.y, 9, 0, Math.PI * 2);
    ctx.fillStyle = cu.col; ctx.fill();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();
  }});
  requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);
</script></body></html>"""


def anim_parallel_html(lanes: list) -> str:
    """
    Animated diagram for n independent parallel lanes.
    Each lane dict: {lam, mu, rho, color, label, spawn_ms}
    Total arrivals across all lanes sum to ~1 customer/second.
    """
    n     = len(lanes)
    H     = n * 120 + 50
    total_lam = sum(la["lam"] for la in lanes)
    scale_s   = round(1 / total_lam, 4) if total_lam > 0 else 1.0

    # Build JS arrays from lanes
    lams      = "[" + ",".join(str(round(la["lam"],  4)) for la in lanes) + "]"
    mus       = "[" + ",".join(str(round(la["mu"],   4)) for la in lanes) + "]"
    rhos      = "[" + ",".join(str(round(la["rho"],  4)) for la in lanes) + "]"
    spawn_mss = "[" + ",".join(str(round(la["spawn_ms"], 1)) for la in lanes) + "]"
    colors    = "[" + ",".join(f'"{la["color"]}"' for la in lanes) + "]"
    labels    = "[" + ",".join(f'"{la["label"]}"' for la in lanes) + "]"

    row_h    = (H - 20) / n
    lane_ys  = "[" + ",".join(str(round(10 + row_h * i + row_h / 2, 1)) for i in range(n)) + "]"

    svc_mss = "[" + ",".join(
        str(round((la["lam"] / la["mu"]) * 1000, 1)) for la in lanes
    ) + "]"

    header_parts = [f'λ{i+1}={la["lam"]:.2f} μ{i+1}={la["mu"]:.2f} ρ{i+1}={la["rho"]:.1%}'
                    for i, la in enumerate(lanes)]
    header = "   |   ".join(header_parts)

    return f"""<!DOCTYPE html><html><head>
<style>
  body {{margin:0;padding:0;background:#fff;font-family:'Inter',sans-serif;}}
  canvas {{display:block;margin:0 auto;}}
  #note {{font-size:10px;color:#94a3b8;text-align:center;padding:3px 0 4px;}}
</style></head><body>
<canvas id="cv" width="760" height="{H}"></canvas>
<div id="note">Animation scale: 1 second ≈ {scale_s} time units &nbsp;|&nbsp; {header}</div>
<script>
const W = 760, H = {H}, N = {n};
const LAMS     = {lams};
const MUS      = {mus};
const RHOS     = {rhos};
const SVC_MSS  = {svc_mss};
const SPAWN_MS = {spawn_mss};
const COLS     = {colors};
const LABELS   = {labels};
const LANE_YS  = {lane_ys};

const Q_END = 330, S_X0 = 390, S_X1 = 570;

// Per-lane state
let servers  = new Array(N).fill(null);
let queues   = Array.from({{length: N}}, () => []);
let lastSpawn = new Array(N).fill(-99999);
let allCusts = [];
let nextId   = 0;

function Customer(lane) {{
  this.id    = nextId++;
  this.lane  = lane;
  this.x     = 5;
  this.y     = LANE_YS[lane];
  this.tx    = Q_END;
  this.ty    = LANE_YS[lane];
  this.state = 'arriving';
  this.sEnd  = 0;
  this.col   = '#f97316';
}}

function dispatchLane(lane, now) {{
  if (!servers[lane] && queues[lane].length > 0) {{
    let cu = queues[lane].shift();
    servers[lane] = cu;
    cu.tx = (S_X0 + S_X1) / 2;
    cu.ty = LANE_YS[lane];
    cu.state = 'to_server';
  }}
}}

function updateCust(cu, dt, now) {{
  let lane = cu.lane;
  switch (cu.state) {{
    case 'arriving':
      cu.x += 180 * dt;
      if (cu.x >= Q_END) {{
        cu.x = Q_END; cu.state = 'queued';
        queues[lane].push(cu); dispatchLane(lane, now);
      }}
      break;
    case 'queued': {{
      let qi = queues[lane].indexOf(cu);
      let tx = Q_END - 22 * (qi + 1);
      cu.x += (tx - cu.x) * 10 * dt;
      break;
    }}
    case 'to_server': {{
      let dx = cu.tx - cu.x, dy = cu.ty - cu.y;
      let d  = Math.sqrt(dx*dx + dy*dy);
      if (d < 3) {{
        cu.x = cu.tx; cu.y = cu.ty;
        cu.state = 'serving';
        cu.sEnd  = now + SVC_MSS[lane];
        cu.col   = '#16a34a';
      }} else {{
        cu.x += dx/d * 250 * dt;
        cu.y += dy/d * 250 * dt;
      }}
      break;
    }}
    case 'serving':
      if (now >= cu.sEnd) {{
        cu.state = 'leaving'; cu.col = '#94a3b8';
        servers[lane] = null;
        dispatchLane(lane, now);
      }}
      break;
    case 'leaving':
      cu.x += 220 * dt;
      if (cu.x > W + 20) cu.state = 'done';
      break;
  }}
}}

function drawStatic(ctx) {{
  for (let i = 0; i < N; i++) {{
    let sy  = LANE_YS[i];
    let col = COLS[i];
    let r = parseInt(col.slice(1,3),16), g = parseInt(col.slice(3,5),16), b = parseInt(col.slice(5,7),16);
    // Arrival line
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(10, sy); ctx.lineTo(Q_END, sy); ctx.stroke();
    // Arrowhead at Q_END
    ctx.fillStyle = '#94a3b8';
    ctx.beginPath(); ctx.moveTo(Q_END, sy);
    ctx.lineTo(Q_END-10, sy-5); ctx.lineTo(Q_END-10, sy+5); ctx.fill();
    // λ label
    ctx.fillStyle = '#374151'; ctx.font = 'bold 11px Inter'; ctx.textAlign = 'left';
    ctx.fillText('λ = ' + LAMS[i].toFixed(2), 12, sy - 12);
    // Queue boundary
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(Q_END, sy - 44); ctx.lineTo(Q_END, sy + 44); ctx.stroke();
    // Connector to server
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(Q_END, sy); ctx.lineTo(S_X0, sy); ctx.stroke();
    // Departure line
    ctx.beginPath(); ctx.moveTo(S_X1, sy); ctx.lineTo(W - 6, sy); ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.beginPath(); ctx.moveTo(W-6, sy);
    ctx.lineTo(W-16, sy-5); ctx.lineTo(W-16, sy+5); ctx.fill();
    // Server box
    ctx.strokeStyle = col; ctx.lineWidth = 2.2;
    ctx.fillStyle   = 'rgba(' + r + ',' + g + ',' + b + ',0.07)';
    ctx.fillRect(S_X0, sy - 28, S_X1 - S_X0, 56);
    ctx.strokeRect(S_X0, sy - 28, S_X1 - S_X0, 56);
    ctx.fillStyle = col; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center';
    ctx.fillText(LABELS[i], (S_X0+S_X1)/2, sy - 7);
    ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter';
    ctx.fillText('μ = ' + MUS[i].toFixed(2) + '   ρ = ' + (RHOS[i]*100).toFixed(1) + '%',
                 (S_X0+S_X1)/2, sy + 11);
  }}
}}

const cv  = document.getElementById('cv');
const ctx = cv.getContext('2d');
let prev  = null;

function loop(ts) {{
  if (!prev) prev = ts;
  let dt = (ts - prev) / 1000; prev = ts;
  dt = Math.min(dt, 0.05);

  for (let i = 0; i < N; i++) {{
    if (ts - lastSpawn[i] >= SPAWN_MS[i]) {{
      allCusts.push(new Customer(i));
      lastSpawn[i] = ts;
    }}
  }}
  allCusts.forEach(cu => updateCust(cu, dt, ts));
  allCusts = allCusts.filter(cu => cu.state !== 'done');

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  drawStatic(ctx);
  allCusts.forEach(cu => {{
    ctx.beginPath(); ctx.arc(cu.x, cu.y, 9, 0, Math.PI*2);
    ctx.fillStyle = cu.col; ctx.fill();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();
  }});
  requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC TABLE  (no traffic intensity row; ρ formula adapts to c)
# ═══════════════════════════════════════════════════════════════════════════════
def metric_table_html(m: dict, color: str = BLUE) -> str:
    rho_formula = "ρ = λ / μ" if m["c"] == 1 else "ρ = λ / (c·μ)"
    rows = [
        ("Arrival rate",              "λ",              fmt(m["lam"], 3)),
        ("Service rate (per server)", "μ",              fmt(m["mu"],  3)),
        ("Number of servers",         "c",              str(m["c"])),
        ("Server utilization",        rho_formula,      fmt(m["rho"], 4, pct=True)),
        ("Prob. system empty",        "P₀",             fmt(m["P0"],  4, pct=True)),
        ("Prob. customer waits",      "C(c, a)",        fmt(m["Pw"],  4, pct=True)),
        ("Avg. customers in queue",   "Lq",             fmt(m["Lq"],  4)),
        ("Avg. customers in service", "Ls = λ / μ",     fmt(m["Ls"],  4)),
        ("Avg. customers in system",  "L = Lq + Ls",    fmt(m["L"],   4)),
        ("Avg. wait in queue",        "Wq",             fmt(m["Wq"],  4, time=True)),
        ("Avg. time in system",       "W = Wq + 1/μ",  fmt(m["W"],   4, time=True)),
    ]
    html = ('<table class="metric-tbl"><tr>'
            '<th>Parameter</th><th>Formula</th><th>Value</th></tr>')
    for param, formula, value in rows:
        hl = (' class="hl"' if param.startswith("Avg.")
              else f' style="color:{color};font-weight:600;"'
              if "Prob." in param else "")
        html += (f'<tr><td>{param}</td>'
                 f'<td style="color:#64748b;font-family:monospace;">{formula}</td>'
                 f'<td{hl}>{value}</td></tr>')
    html += "</table>"
    return html


def kpi(label, value, color):
    return (f'<div class="kpi-card" style="border-top-color:{color};">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-val" style="color:{color};">{value}</div></div>')


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="page-title">BAT 3301 – Colazo – Waiting Lines</div>'
    '<div class="page-sub">M/M/c queuing theory · interactive demo · '
    'Erlang-C model · Poisson arrivals · exponential service</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "📊  M/M/1 to M/M/4",
    "🔀  n × M/M/1  vs  M/M/n",
    "🚀  Express Lines",
])

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — M/M/c  (c = 1 .. 4)                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="sec-hdr">Parameters</div>', unsafe_allow_html=True)
    inp_col, _, metric_col = st.columns([1.2, 0.08, 2.5])

    with inp_col:
        mode1 = st.radio("Input mode", ["λ and μ", "ρ and μ"],
                         horizontal=True, label_visibility="collapsed")
        c_sel = st.selectbox("Number of servers (c)", [1, 2, 3, 4], index=0)

        if mode1 == "λ and μ":
            lam1 = st.number_input("Arrival rate  λ (customers/unit time)",
                                   value=3.0, min_value=0.01, step=0.5, format="%.2f")
            mu1  = st.number_input("Service rate  μ (customers/unit time)",
                                   value=4.0, min_value=0.01, step=0.5, format="%.2f")
        else:
            rho1_in = st.slider("Per-server utilization  ρ", 0.01, 0.99, 0.60, 0.01)
            mu1     = st.number_input("Service rate  μ (customers/unit time)",
                                      value=4.0, min_value=0.01, step=0.5, format="%.2f")
            lam1 = rho1_in * c_sel * mu1
            st.caption(f"→ λ = ρ × c × μ = {rho1_in:.2f} × {c_sel} × {mu1:.2f} = **{lam1:.3f}**")

        m1 = mmc(lam1, mu1, c_sel)

    with metric_col:
        if m1 is None:
            st.markdown(
                f'<div class="warn-box">⚠ <b>System unstable</b>: '
                f'ρ = λ/(c·μ) = {lam1:.2f}/({c_sel}×{mu1:.2f}) = '
                f'{lam1/(c_sel*mu1):.3f} ≥ 1. '
                f'Reduce λ, increase μ, or add more servers.</div>',
                unsafe_allow_html=True,
            )
        else:
            color1 = SERVER_COLORS[(c_sel - 1) % len(SERVER_COLORS)]
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.markdown(kpi("Utilization ρ", f"{m1['rho']:.1%}", color1),
                            unsafe_allow_html=True)
            with kpi_cols[1]:
                st.markdown(kpi("Avg. queue  Lq", f"{m1['Lq']:.3f}", ORANGE),
                            unsafe_allow_html=True)
            with kpi_cols[2]:
                st.markdown(kpi("Avg. system  L", f"{m1['L']:.3f}", GREEN),
                            unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(kpi("Wait in queue Wq", f"{m1['Wq']:.4f}", RED),
                            unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(metric_table_html(m1, color1), unsafe_allow_html=True)

    if m1 is not None:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Queue Diagram — Dynamic Simulation</div>',
                    unsafe_allow_html=True)
        components.html(
            anim_mmc_html(m1["lam"], m1["mu"], c_sel, m1["rho"]),
            height=max(c_sel, 2) * 110 + 80,
            scrolling=False,
        )

        # ── All-c comparison ──────────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Comparison across c = 1, 2, 3, 4  '
                    '(same λ and μ)</div>', unsafe_allow_html=True)

        comp_rows = []
        for c in [1, 2, 3, 4]:
            mc = mmc(lam1, mu1, c)
            if mc:
                comp_rows.append({
                    "c": c, "ρ": f"{mc['rho']:.1%}",
                    "P₀": f"{mc['P0']:.4f}", "C(c,a)": f"{mc['Pw']:.4f}",
                    "Lq": f"{mc['Lq']:.4f}", "L": f"{mc['L']:.4f}",
                    "Wq": f"{mc['Wq']:.4f}", "W": f"{mc['W']:.4f}",
                })
            else:
                comp_rows.append({
                    "c": c, "ρ": f"{lam1/(c*mu1):.3f} ⚠",
                    "P₀": "unstable", "C(c,a)": "—",
                    "Lq": "—", "L": "—", "Wq": "—", "W": "—",
                })

        df_comp = pd.DataFrame(comp_rows).set_index("c")

        def _hl_row(row):
            if row.name == c_sel:
                return [f"background:{hex_rgba(color1,0.15)}; "
                        f"color:{color1}; font-weight:700;"] * len(row)
            return [""] * len(row)

        st.dataframe(df_comp.style.apply(_hl_row, axis=1),
                     use_container_width=True, height=190)
        st.caption(f"▶ Highlighted row = current selection (c = {c_sel}). "
                   "Adding servers reduces Lq, L, Wq, and W dramatically.")

        valid_cs = [r["c"] for r in comp_rows if r["Lq"] != "—"]
        Lq_vals  = [mmc(lam1, mu1, c)["Lq"] for c in valid_cs]
        L_vals   = [mmc(lam1, mu1, c)["L"]  for c in valid_cs]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=[f"c={c}" for c in valid_cs], y=Lq_vals,
            name="Lq (in queue)", marker_color=ORANGE, opacity=0.85))
        fig_bar.add_trace(go.Bar(x=[f"c={c}" for c in valid_cs], y=L_vals,
            name="L (in system)", marker_color=BLUE, opacity=0.85))
        fig_bar.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            barmode="group", height=260,
            margin=dict(t=14, b=40, l=50, r=20),
            xaxis=dict(title="Number of servers (c)",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            yaxis=dict(title="Average customers",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_bar, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Lq and Wq vs ρ curves ─────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-hdr">Lq and Wq as a function of ρ  '
            '— all c values (μ held constant)</div>',
            unsafe_allow_html=True,
        )

        rho_sweep = np.linspace(0.01, 0.995, 300)
        _curve_layout = dict(
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            height=290,
            margin=dict(t=14, b=52, l=62, r=24),
            xaxis=dict(
                title="Per-server utilization  ρ",
                range=[0, 1],
                gridcolor="#f1f5f9", linecolor="#e2e8f0",
                tickformat=".0%",
                tickfont=dict(size=11),
                title_font=dict(size=13),
            ),
            yaxis=dict(
                gridcolor="#f1f5f9", linecolor="#e2e8f0",
                tickfont=dict(size=11),
                title_font=dict(size=13),
            ),
            legend=dict(
                orientation="h", y=1.08, x=0.5, xanchor="center",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0", borderwidth=1,
            ),
        )

        fig_lq_rho = go.Figure()
        fig_wq_rho = go.Figure()

        for c in [1, 2, 3, 4]:
            col_c  = SERVER_COLORS[(c - 1) % len(SERVER_COLORS)]
            is_sel = (c == c_sel)
            lq_pts, wq_pts, rp_pts = [], [], []
            for rho_v in rho_sweep:
                lam_v = rho_v * c * mu1
                m_v   = mmc(lam_v, mu1, c)
                if m_v:
                    lq_pts.append(m_v["Lq"])
                    wq_pts.append(m_v["Wq"])
                    rp_pts.append(rho_v)

            line_style = dict(
                color=col_c,
                width=3.0 if is_sel else 1.6,
                dash="solid" if is_sel else "dot",
            )
            label = f"c = {c}  ◀ selected" if is_sel else f"c = {c}"
            # Lq curve
            fig_lq_rho.add_trace(go.Scatter(
                x=rp_pts, y=lq_pts, mode="lines", name=label,
                line=line_style,
                hovertemplate=(f"c={c}<br>ρ = %{{x:.1%}}"
                               f"<br>Lq = %{{y:.4f}}<extra></extra>"),
            ))
            # Wq curve
            fig_wq_rho.add_trace(go.Scatter(
                x=rp_pts, y=wq_pts, mode="lines", name=label,
                line=line_style,
                hovertemplate=(f"c={c}<br>ρ = %{{x:.1%}}"
                               f"<br>Wq = %{{y:.4f}}<extra></extra>"),
            ))

        # Current operating point markers
        fig_lq_rho.add_trace(go.Scatter(
            x=[m1["rho"]], y=[m1["Lq"]],
            mode="markers", name=f"Current  (ρ={m1['rho']:.2%})",
            marker=dict(color=RED, size=11, symbol="circle",
                        line=dict(color="white", width=2)),
            hovertemplate=(f"Operating point<br>ρ={m1['rho']:.4f}"
                           f"<br>Lq={m1['Lq']:.4f}<extra></extra>"),
        ))
        fig_wq_rho.add_trace(go.Scatter(
            x=[m1["rho"]], y=[m1["Wq"]],
            mode="markers", name=f"Current  (ρ={m1['rho']:.2%})",
            marker=dict(color=RED, size=11, symbol="circle",
                        line=dict(color="white", width=2)),
            hovertemplate=(f"Operating point<br>ρ={m1['rho']:.4f}"
                           f"<br>Wq={m1['Wq']:.4f}<extra></extra>"),
        ))

        # Vertical line at current ρ
        for _fig in [fig_lq_rho, fig_wq_rho]:
            _fig.add_vline(
                x=m1["rho"],
                line_color=RED, line_width=1.2, line_dash="dot",
                annotation_text=f"  ρ = {m1['rho']:.2%}",
                annotation_font=dict(size=10, color=RED),
                annotation_position="top right",
            )

        fig_lq_rho.update_layout(
            **_curve_layout,
            yaxis_title="Avg. queue length  Lq",
        )
        fig_wq_rho.update_layout(
            **_curve_layout,
            yaxis_title="Avg. wait in queue  Wq",
        )

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(fig_lq_rho, use_container_width=True,
                            config={"displayModeBar": False})
        with ch2:
            st.plotly_chart(fig_wq_rho, use_container_width=True,
                            config={"displayModeBar": False})

        st.caption(
            f"Curves show how Lq and Wq grow as per-server utilization ρ → 1, "
            f"holding μ = {mu1:.2f} fixed (λ varies to achieve each ρ).  "
            f"Solid line = selected c ({c_sel}).  "
            f"Red dot = current operating point  "
            f"(ρ = {m1['rho']:.1%},  Lq = {m1['Lq']:.4f},  Wq = {m1['Wq']:.4f})."
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — n × M/M/1  vs  M/M/n                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="sec-hdr">Parameters</div>', unsafe_allow_html=True)
    ip2_a, ip2_b, ip2_c = st.columns([1, 1, 1])
    with ip2_a:
        lam2 = st.number_input("Total arrival rate  λ", value=6.0,
                               min_value=0.01, step=0.5, format="%.2f", key="lam2")
    with ip2_b:
        mu2  = st.number_input("Service rate per server  μ", value=4.0,
                               min_value=0.01, step=0.5, format="%.2f", key="mu2")
    with ip2_c:
        n2   = st.selectbox("Number of servers / queues  (n)", [1, 2, 3, 4],
                            index=1, key="n2")

    m2_each = mmc(lam2 / n2, mu2, 1)
    m2_mmn  = mmc(lam2, mu2, n2)

    if m2_each is None or m2_mmn is None:
        rho_val = lam2 / (n2 * mu2)
        st.markdown(
            f'<div class="warn-box">⚠ System unstable: '
            f'ρ = {lam2:.2f}/({n2}×{mu2:.2f}) = {rho_val:.3f} ≥ 1</div>',
            unsafe_allow_html=True,
        )
    else:
        total_L_nxmm1  = n2 * m2_each["L"]
        total_Lq_nxmm1 = n2 * m2_each["Lq"]

        head_a, head_b = st.columns(2)
        with head_a:
            st.markdown(
                f'<div class="info-box" style="border-left-color:{ORANGE};">'
                f'<b style="color:{ORANGE};">{n2} × M/M/1 — {n2} separate queues</b><br>'
                f'Each queue receives λ/n = {lam2:.2f}/{n2} = {lam2/n2:.3f} '
                f'arrivals/unit time. Customers join one queue and '
                f'<em>cannot switch</em> even if another is shorter.</div>',
                unsafe_allow_html=True,
            )
        with head_b:
            st.markdown(
                f'<div class="info-box" style="border-left-color:{BLUE};">'
                f'<b style="color:{BLUE};">M/M/{n2} — single shared queue</b><br>'
                f'All {lam2:.2f} arrivals join one queue; the next free server '
                f'takes the front customer. Always at least as efficient as {n2}×M/M/1.</div>',
                unsafe_allow_html=True,
            )

        # ── Animated diagrams ─────────────────────────────────────────────────
        diag_a, diag_b = st.columns(2)

        # n×M/M/1 lanes (total = 1 cust/sec → each lane spawns every n*1000ms)
        nxmm1_lanes = [
            {
                "lam":      round(lam2 / n2, 4),
                "mu":       mu2,
                "rho":      m2_each["rho"],
                "color":    LANE_COLORS[i % len(LANE_COLORS)],
                "label":    f"Queue {i+1} / Server {i+1}",
                "spawn_ms": 1000 * n2,   # spread total 1/sec across n lanes
            }
            for i in range(n2)
        ]

        with diag_a:
            st.markdown(f'<div class="sec-hdr">{n2} × M/M/1 — Dynamic Simulation</div>',
                        unsafe_allow_html=True)
            components.html(
                anim_parallel_html(nxmm1_lanes),
                height=n2 * 120 + 80,
                scrolling=False,
            )

        with diag_b:
            st.markdown(f'<div class="sec-hdr">M/M/{n2} — Dynamic Simulation</div>',
                        unsafe_allow_html=True)
            components.html(
                anim_mmc_html(m2_mmn["lam"], m2_mmn["mu"], n2, m2_mmn["rho"]),
                height=max(n2, 2) * 110 + 80,
                scrolling=False,
            )

        # ── Comparison table ──────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Side-by-side comparison</div>',
                    unsafe_allow_html=True)

        metrics_list = [
            ("Utilization per server",       "ρ",
             f"{m2_each['rho']:.4f}", f"{m2_mmn['rho']:.4f}"),
            ("Prob. system empty",           "P₀",
             f"{m2_each['P0']:.4f}", f"{m2_mmn['P0']:.4f}"),
            ("Prob. customer waits",         "C(c,a)",
             f"{m2_each['Pw']:.4f}", f"{m2_mmn['Pw']:.4f}"),
            ("Avg. customers in queue (each)","Lq",
             f"{m2_each['Lq']:.4f}", f"{m2_mmn['Lq']:.4f}"),
            ("Avg. customers in system (each)","L",
             f"{m2_each['L']:.4f}", f"{m2_mmn['L']:.4f}"),
            ("★ Total customers in system",  "L_total",
             f"{total_L_nxmm1:.4f}", f"{m2_mmn['L']:.4f}"),
            ("★ Total customers in queue",   "Lq_total",
             f"{total_Lq_nxmm1:.4f}", f"{m2_mmn['Lq']:.4f}"),
            ("Avg. wait in queue",           "Wq",
             f"{m2_each['Wq']:.4f}", f"{m2_mmn['Wq']:.4f}"),
            ("Avg. time in system",          "W",
             f"{m2_each['W']:.4f}", f"{m2_mmn['W']:.4f}"),
        ]

        tbl = (f'<table class="cmp-tbl"><tr><th>Metric</th><th>Formula</th>'
               f'<th style="color:{ORANGE};">{n2} × M/M/1</th>'
               f'<th style="color:{BLUE};">M/M/{n2}</th>'
               f'<th>Better?</th></tr>')
        for param, formula, v_nx, v_mmn in metrics_list:
            try:
                a_v, b_v = float(v_nx), float(v_mmn)
                if b_v < a_v:
                    cls_a, cls_b = " class='worse'", " class='better'"
                    better = f'<span style="color:{BLUE}">M/M/{n2} ✓</span>'
                elif a_v < b_v:
                    cls_a, cls_b = " class='better'", " class='worse'"
                    better = f'<span style="color:{ORANGE}">{n2}×M/M/1 ✓</span>'
                else:
                    cls_a = cls_b = ""; better = "Equal"
            except Exception:
                cls_a = cls_b = ""; better = "—"
            bold = " style='font-weight:700;'" if param.startswith("★") else ""
            tbl += (f'<tr><td{bold}>{param}</td>'
                    f'<td style="color:#64748b;font-family:monospace;">{formula}</td>'
                    f'<td{cls_a}>{v_nx}</td><td{cls_b}>{v_mmn}</td>'
                    f'<td style="font-size:12px;">{better}</td></tr>')
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

        imp_Lq = (total_Lq_nxmm1 - m2_mmn["Lq"]) / total_Lq_nxmm1 * 100
        imp_Wq = (m2_each["Wq"]  - m2_mmn["Wq"])  / m2_each["Wq"]  * 100
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box" style="border-left-color:{GREEN};">'
            f'<b style="color:{GREEN};">Pooling effect:</b> '
            f'The M/M/{n2} system reduces total queue length (Lq) by '
            f'<b>{imp_Lq:.1f}%</b> and waiting time (Wq) by '
            f'<b>{imp_Wq:.1f}%</b> vs {n2} separate M/M/1 queues.</div>',
            unsafe_allow_html=True,
        )

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            x=[f"{n2}×M/M/1", f"M/M/{n2}"],
            y=[total_Lq_nxmm1, m2_mmn["Lq"]],
            marker_color=[ORANGE, BLUE], opacity=0.85,
            text=[f"{total_Lq_nxmm1:.3f}", f"{m2_mmn['Lq']:.3f}"],
            textposition="outside",
        ))
        fig_cmp.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            height=260, showlegend=False,
            margin=dict(t=14, b=40, l=60, r=20),
            yaxis=dict(title="Total customers in queue (Lq)",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True,
                        config={"displayModeBar": False})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — Express Lines                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="sec-hdr">Express Line System Parameters</div>',
                unsafe_allow_html=True)
    c3_a, c3_b, c3_c, c3_d = st.columns(4)
    with c3_a:
        lam3_total = st.number_input("Total arrival rate  λ", value=5.0,
                                     min_value=0.01, step=0.5, format="%.2f", key="lam3")
    with c3_b:
        pct3 = st.slider("% routed to express lane",
                         min_value=5, max_value=95, value=40, step=5,
                         format="%d%%") / 100
    with c3_c:
        mu3_fast = st.number_input("Express server rate  μ₁ (fast)",
                                   value=6.0, min_value=0.01, step=0.5,
                                   format="%.2f", key="mu3f")
    with c3_d:
        mu3_slow = st.number_input("Regular server rate  μ₂ (slow)",
                                   value=5.0, min_value=0.01, step=0.5,
                                   format="%.2f", key="mu3s")

    lam3_fast = pct3 * lam3_total
    lam3_slow = (1 - pct3) * lam3_total
    m3f = mmc(lam3_fast, mu3_fast, 1)
    m3s = mmc(lam3_slow, mu3_slow, 1)

    if m3f is None or m3s is None:
        msgs = []
        if m3f is None:
            msgs.append(f"Express lane unstable: ρ₁ = {lam3_fast:.2f}/{mu3_fast:.2f} = "
                        f"{lam3_fast/mu3_fast:.3f} ≥ 1. Increase μ₁ or reduce λ.")
        if m3s is None:
            msgs.append(f"Regular lane unstable: ρ₂ = {lam3_slow:.2f}/{mu3_slow:.2f} = "
                        f"{lam3_slow/mu3_slow:.3f} ≥ 1. Increase μ₂ or reduce λ.")
        for msg in msgs:
            st.markdown(f'<div class="warn-box">⚠ {msg}</div>',
                        unsafe_allow_html=True)
    else:
        L_total  = m3f["L"]  + m3s["L"]
        Lq_total = m3f["Lq"] + m3s["Lq"]
        Wq_avg   = (lam3_fast * m3f["Wq"] + lam3_slow * m3s["Wq"]) / lam3_total
        W_avg    = (lam3_fast * m3f["W"]  + lam3_slow * m3s["W"])  / lam3_total

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(kpi("Total L (system)", f"{L_total:.3f}", BLUE),
                        unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(kpi("Total Lq (queue)", f"{Lq_total:.3f}", ORANGE),
                        unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(kpi("Weighted Wq (avg)", f"{Wq_avg:.4f}", RED),
                        unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(kpi("Weighted W (avg)", f"{W_avg:.4f}", GREEN),
                        unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Animated diagram ──────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">System Diagram — Dynamic Simulation</div>',
                    unsafe_allow_html=True)

        express_lanes = [
            {
                "lam":      round(lam3_fast, 4),
                "mu":       mu3_fast,
                "rho":      m3f["rho"],
                "color":    BLUE,
                "label":    f"Express Server  ({pct3:.0%})",
                "spawn_ms": round(1000 / pct3, 1),   # proportional to split
            },
            {
                "lam":      round(lam3_slow, 4),
                "mu":       mu3_slow,
                "rho":      m3s["rho"],
                "color":    ORANGE,
                "label":    f"Regular Server  ({1-pct3:.0%})",
                "spawn_ms": round(1000 / (1 - pct3), 1),
            },
        ]
        components.html(
            anim_parallel_html(express_lanes),
            height=2 * 120 + 80,
            scrolling=False,
        )

        # ── Per-lane metrics ──────────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        lane_a, lane_b = st.columns(2)
        with lane_a:
            st.markdown(f'<div class="sec-hdr">Express Lane '
                        f'({pct3:.0%} of traffic)</div>', unsafe_allow_html=True)
            st.markdown(metric_table_html(m3f, BLUE), unsafe_allow_html=True)
        with lane_b:
            st.markdown(f'<div class="sec-hdr">Regular Lane '
                        f'({1-pct3:.0%} of traffic)</div>', unsafe_allow_html=True)
            st.markdown(metric_table_html(m3s, ORANGE), unsafe_allow_html=True)

        # ── System summary ────────────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">System-Level Summary</div>',
                    unsafe_allow_html=True)
        sys_rows = [
            ("Total arrival rate",       "λ",            f"{lam3_total:.3f}"),
            ("Express lane arrivals",    "λ₁ = p·λ",     f"{lam3_fast:.3f} ({pct3:.0%})"),
            ("Regular lane arrivals",    "λ₂ = (1−p)·λ", f"{lam3_slow:.3f} ({1-pct3:.0%})"),
            ("Express server rate",      "μ₁",           f"{mu3_fast:.3f}"),
            ("Regular server rate",      "μ₂",           f"{mu3_slow:.3f}"),
            ("Express utilization",      "ρ₁ = λ₁/μ₁",  f"{m3f['rho']:.4f} = {m3f['rho']:.1%}"),
            ("Regular utilization",      "ρ₂ = λ₂/μ₂",  f"{m3s['rho']:.4f} = {m3s['rho']:.1%}"),
            ("Total Lq (both lanes)",    "Lq₁ + Lq₂",   f"{Lq_total:.4f}"),
            ("Total L  (both lanes)",    "L₁ + L₂",      f"{L_total:.4f}"),
            ("Weighted avg. Wq",         "Σλᵢ·Wqᵢ / λ", f"{Wq_avg:.4f}"),
            ("Weighted avg. W",          "Σλᵢ·Wᵢ / λ",  f"{W_avg:.4f}"),
        ]
        tbl3 = ('<table class="metric-tbl"><tr>'
                '<th>Parameter</th><th>Formula</th><th>Value</th></tr>')
        for p, f_, v in sys_rows:
            bold = " style='font-weight:700;'" if "Total" in p or "Weighted" in p else ""
            tbl3 += (f'<tr><td{bold}>{p}</td>'
                     f'<td style="color:#64748b;font-family:monospace;">{f_}</td>'
                     f'<td{bold}>{v}</td></tr>')
        tbl3 += "</table>"
        st.markdown(tbl3, unsafe_allow_html=True)

        # ── Sensitivity curve ─────────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Sensitivity — Wq vs. Express Split %</div>',
                    unsafe_allow_html=True)
        splits   = np.linspace(0.05, 0.95, 80)
        wq_curve = []
        for p in splits:
            mf_ = mmc(p * lam3_total, mu3_fast, 1)
            ms_ = mmc((1 - p) * lam3_total, mu3_slow, 1)
            if mf_ and ms_:
                wq_curve.append(
                    (p * lam3_total * mf_["Wq"] +
                     (1 - p) * lam3_total * ms_["Wq"]) / lam3_total
                )
            else:
                wq_curve.append(None)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=splits * 100, y=wq_curve, mode="lines",
            line=dict(color=BLUE, width=2.5),
            hovertemplate="Split: %{x:.0f}%<br>Wq: %{y:.4f}<extra></extra>",
        ))
        fig_sens.add_vline(x=pct3 * 100,
            line_color=RED, line_width=1.5, line_dash="dash",
            annotation_text=f"  Current: {pct3:.0%}",
            annotation_font=dict(color=RED, size=11))
        fig_sens.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            height=270, showlegend=False,
            margin=dict(t=14, b=48, l=60, r=20),
            xaxis=dict(title="% routed to express lane",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            yaxis=dict(title="Weighted average Wq",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        )
        st.plotly_chart(fig_sens, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "Curve: weighted average Wq across all express split percentages. "
            "Red line: current setting. Gaps = unstable configurations (ρ ≥ 1). "
            "The minimum of the curve is the optimal split."
        )
