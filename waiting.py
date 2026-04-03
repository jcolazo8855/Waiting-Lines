# ═══════════════════════════════════════════════════════════════════════════════
#  BAT 3301 – Colazo – Waiting Lines
#  M/M/c queues · Shewhart · 3-sigma limits · Interactive Streamlit demo
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
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

/* Tabs */
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

/* Page title */
.page-title {
    font-size: 26px; font-weight: 800; letter-spacing: -0.8px;
    color: #0f172a; margin-bottom: 2px;
}
.page-sub { font-size: 13px; color: #94a3b8; margin-bottom: 20px; }

/* Section header */
.sec-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    color: #94a3b8; text-transform: uppercase;
    border-bottom: 1px solid #e2e8f0; padding-bottom: 7px; margin-bottom: 14px;
}

/* KPI cards */
.kpi-card {
    background: #f8f9fc; border: 1px solid #e2e8f0;
    border-top: 3px solid; border-radius: 6px; padding: 12px 14px;
}
.kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    color: #94a3b8; text-transform: uppercase;
}
.kpi-val { font-size: 22px; font-weight: 700; margin-top: 5px; letter-spacing: -0.5px; }

/* Info box */
.info-box {
    background: #f8f9fc; border: 1px solid #e2e8f0; border-left: 4px solid;
    border-radius: 0 6px 6px 0; padding: 12px 16px;
    font-size: 13px; line-height: 1.75; color: #374151; margin-bottom: 12px;
}

/* Warning / error */
.warn-box {
    background: #fef2f2; border: 1px solid #fca5a5; border-radius: 6px;
    padding: 12px 16px; font-size: 13px; color: #7f1d1d;
}

/* Metric table */
.metric-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.metric-tbl th {
    background:#f1f5f9; text-align:left; padding:9px 12px;
    font-weight:600; color:#374151; border:1px solid #e2e8f0; font-size:12px;
}
.metric-tbl td { padding:8px 12px; border:1px solid #e2e8f0; color:#374151; }
.metric-tbl tr:nth-child(even) td { background:#f8f9fc; }
.metric-tbl td.hl { font-weight:700; }

/* Compare table */
.cmp-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.cmp-tbl th { background:#f1f5f9; text-align:center; padding:9px 12px; font-weight:600; border:1px solid #e2e8f0; }
.cmp-tbl td { padding:8px 12px; border:1px solid #e2e8f0; text-align:center; }
.cmp-tbl tr:nth-child(even) td { background:#f8f9fc; }
.cmp-tbl td.better { color:#16a34a; font-weight:700; }
.cmp-tbl td.worse  { color:#dc2626; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
BLUE    = "#2563eb"
GREEN   = "#16a34a"
RED     = "#dc2626"
ORANGE  = "#f97316"
PURPLE  = "#7c3aed"
TEAL    = "#0891b2"
AMBER   = "#d97706"
GRAY    = "#6b7280"

SERVER_COLORS = [BLUE, GREEN, PURPLE, TEAL]   # one per server in diagrams
LANE_COLORS   = [BLUE, ORANGE, GREEN, PURPLE]  # one per parallel lane

def hex_rgba(h, a=0.12):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

# ═══════════════════════════════════════════════════════════════════════════════
#  M/M/c MATH
# ═══════════════════════════════════════════════════════════════════════════════
def mmc(lam: float, mu: float, c: int) -> dict | None:
    """Compute all M/M/c steady-state metrics. Returns None if unstable."""
    if lam <= 0 or mu <= 0 or c < 1:
        return None
    a   = lam / mu          # traffic intensity (Erlang)
    rho = a / c             # per-server utilisation
    if rho >= 1.0:
        return None

    # P₀: probability system empty
    s    = sum(a**n / math.factorial(n) for n in range(c))
    last = a**c / (math.factorial(c) * (1 - rho))
    P0   = 1.0 / (s + last)

    # Erlang-C: probability arriving customer must wait
    Pw = last * P0

    # Average customers in queue / system
    Lq = Pw * rho / (1 - rho)
    Wq = Lq / lam
    W  = Wq + 1.0 / mu
    L  = lam * W          # = Lq + a
    Ls = a                # in service = λ/μ

    return dict(lam=lam, mu=mu, c=c, a=a, rho=rho,
                P0=P0, Pw=Pw, Lq=Lq, L=L, Ls=Ls, Wq=Wq, W=W)


def fmt(v, decimals=4, pct=False, time=False):
    """Format a metric value for display."""
    if v is None: return "—"
    if pct:   return f"{v:.2%}"
    if time:  return f"{v:.4f}"
    return f"{v:.{decimals}f}"

# ═══════════════════════════════════════════════════════════════════════════════
#  QUEUE DIAGRAM BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
_FONT = dict(family="Inter, sans-serif", color="#374151")
_RH   = 0.72    # row height
_GAP  = 0.35    # gap between rows


def _base_fig(n_rows: int, extra_height: int = 0) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        xaxis=dict(visible=False, range=[-0.2, 10.5]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1,
                   range=[-(n_rows)*(_RH+_GAP)/2 - 0.6,
                           (n_rows)*(_RH+_GAP)/2 + 0.6]),
        height=max(220, 110 * n_rows + extra_height),
        margin=dict(t=8, b=8, l=8, r=8),
        showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                    font=dict(size=12, color="#374151"),
                    bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def _row_y(i: int, n: int) -> float:
    """y-centre of row i (0-indexed) for n total rows."""
    return (n - 1) / 2 * (_RH + _GAP) - i * (_RH + _GAP)


def _add_lane(fig, y, lam, mu, rho, Lq, label, color,
              show_wait_legend=False, show_serve_legend=False,
              queue_x_end=4.5, server_x0=5.2, server_x1=7.8,
              departure_x=9.0):
    """Draw one complete queue-lane at vertical position y."""
    nw = min(max(round(Lq + 0.3), 0), 10)  # customers to show in queue
    ns = rho > 0.05                          # show customer in server?

    # ── arrival arrow ──────────────────────────────────────────────────────
    fig.add_annotation(
        x=1.2, y=y, ax=0.05, ay=y,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"<b>λ={lam:.2f}</b>", showarrow=True,
        arrowhead=3, arrowwidth=2.5, arrowcolor=GRAY,
        font=dict(size=10, color=GRAY),
        xanchor="right", yanchor="middle",
    )

    # ── queue line ─────────────────────────────────────────────────────────
    fig.add_shape(type="line",
        x0=1.25, x1=queue_x_end + 0.1, y0=y, y1=y,
        line_color="#cbd5e1", line_width=1.8)

    # ── waiting customers ──────────────────────────────────────────────────
    if nw > 0:
        qx = [queue_x_end - 0.4 * i for i in range(nw)]
        fig.add_trace(go.Scatter(
            x=qx, y=[y] * nw, mode="markers",
            marker=dict(size=14, color=ORANGE,
                        line=dict(color="white", width=2.5), symbol="circle"),
            name="Waiting" if show_wait_legend else None,
            showlegend=show_wait_legend,
            legendgroup="wait",
            hovertemplate=f"Waiting customers: {nw} (Lq≈{Lq:.2f})<extra></extra>",
        ))

    # ── queue boundary tick ─────────────────────────────────────────────────
    fig.add_shape(type="line",
        x0=queue_x_end + 0.15, x1=queue_x_end + 0.15,
        y0=y - _RH/2 + 0.05, y1=y + _RH/2 - 0.05,
        line_color="#94a3b8", line_width=1.5)

    # ── connector line queue→server ─────────────────────────────────────────
    fig.add_shape(type="line",
        x0=queue_x_end + 0.15, x1=server_x0,
        y0=y, y1=y,
        line_color="#94a3b8", line_width=1.5)

    # ── server box ─────────────────────────────────────────────────────────
    fig.add_shape(type="rect",
        x0=server_x0, x1=server_x1,
        y0=y - _RH/2 + 0.04, y1=y + _RH/2 - 0.04,
        fillcolor=hex_rgba(color, 0.10),
        line_color=color, line_width=2.5)

    sx = (server_x0 + server_x1) / 2
    fig.add_annotation(x=sx, y=y + 0.06,
        text=f"<b>{label}</b>",
        showarrow=False, font=dict(size=11, color=color))
    fig.add_annotation(x=sx, y=y - 0.18,
        text=f"μ = {mu:.2f} | ρ = {rho:.1%}",
        showarrow=False, font=dict(size=9.5, color=GRAY))

    # ── customer in service ────────────────────────────────────────────────
    if ns:
        fig.add_trace(go.Scatter(
            x=[server_x0 + 0.38], y=[y], mode="markers",
            marker=dict(size=14, color=GREEN,
                        line=dict(color="white", width=2.5), symbol="circle"),
            name="In service" if show_serve_legend else None,
            showlegend=show_serve_legend,
            legendgroup="serve",
            hovertemplate=f"In service (ρ={rho:.1%})<extra></extra>",
        ))

    # ── departure arrow ─────────────────────────────────────────────────────
    fig.add_annotation(
        x=departure_x, y=y, ax=server_x1 + 0.05, ay=y,
        xref="x", yref="y", axref="x", ayref="y",
        text="", showarrow=True,
        arrowhead=3, arrowwidth=2.5, arrowcolor=GRAY,
    )


def draw_mmc(m: dict) -> go.Figure:
    """Single queue feeding c servers (M/M/c)."""
    c   = m["c"]
    Lq  = m["Lq"]
    rho = m["rho"]
    lam = m["lam"]
    mu  = m["mu"]

    fig = _base_fig(max(c, 2))

    # Arrival arrow (single, at vertical centre)
    fig.add_annotation(
        x=1.2, y=0, ax=0.05, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"<b>λ={lam:.2f}</b>", showarrow=True,
        arrowhead=3, arrowwidth=2.5, arrowcolor=GRAY,
        font=dict(size=10, color=GRAY),
        xanchor="right", yanchor="middle",
    )

    # Queue line & waiting customers (all in the single queue at y=0)
    nw = min(max(round(Lq + 0.3), 0), 10)
    fig.add_shape(type="line",
        x0=1.25, x1=4.7, y0=0, y1=0,
        line_color="#cbd5e1", line_width=1.8)
    if nw > 0:
        qx = [4.6 - 0.38 * i for i in range(nw)]
        fig.add_trace(go.Scatter(
            x=qx, y=[0] * nw, mode="markers",
            marker=dict(size=14, color=ORANGE,
                        line=dict(color="white", width=2.5)),
            name="Waiting", legendgroup="wait",
            hovertemplate=f"Waiting (Lq≈{Lq:.2f})<extra></extra>",
        ))

    # Queue boundary
    fig.add_shape(type="line",
        x0=4.75, x1=4.75,
        y0=-(c * (_RH + _GAP) / 2) + 0.1,
        y1= (c * (_RH + _GAP) / 2) - 0.1,
        line_color="#94a3b8", line_width=1.5)

    # Fan-out lines from queue to each server
    for i in range(c):
        ys = _row_y(i, c)
        col = SERVER_COLORS[i % len(SERVER_COLORS)]
        fig.add_shape(type="line",
            x0=4.75, x1=5.4, y0=0, y1=ys,
            line_color="#94a3b8", line_width=1.2, line_dash="dot")
        # Server box
        fig.add_shape(type="rect",
            x0=5.4, x1=8.0,
            y0=ys - _RH/2 + 0.04, y1=ys + _RH/2 - 0.04,
            fillcolor=hex_rgba(col, 0.10),
            line_color=col, line_width=2.5)
        sx = 6.7
        fig.add_annotation(x=sx, y=ys + 0.07,
            text=f"<b>Server {i+1}</b>",
            showarrow=False, font=dict(size=11, color=col))
        fig.add_annotation(x=sx, y=ys - 0.18,
            text=f"μ = {mu:.2f} | ρ = {rho:.1%}",
            showarrow=False, font=dict(size=9.5, color=GRAY))
        # Customer in service
        if rho > 0.05:
            fig.add_trace(go.Scatter(
                x=[5.75], y=[ys], mode="markers",
                marker=dict(size=14, color=GREEN,
                            line=dict(color="white", width=2.5)),
                name="In service" if i == 0 else None,
                showlegend=(i == 0),
                legendgroup="serve",
                hovertemplate=f"Server {i+1} (ρ={rho:.1%})<extra></extra>",
            ))
        # Departure arrow
        fig.add_annotation(
            x=9.0, y=ys, ax=8.05, ay=ys,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=3, arrowwidth=2.5, arrowcolor=GRAY,
        )

    return fig


def draw_nxmm1(n: int, m_each: dict) -> go.Figure:
    """n parallel M/M/1 queues (each gets λ/n)."""
    fig = _base_fig(n)
    for i in range(n):
        y   = _row_y(i, n)
        col = LANE_COLORS[i % len(LANE_COLORS)]
        _add_lane(fig, y,
                  lam=m_each["lam"], mu=m_each["mu"],
                  rho=m_each["rho"], Lq=m_each["Lq"],
                  label=f"Queue {i+1} / Server {i+1}", color=col,
                  show_wait_legend=(i == 0),
                  show_serve_legend=(i == 0))
    return fig


def draw_mmn(m: dict) -> go.Figure:
    """M/M/n — same drawing as M/M/c but labeled as 'consolidated queue'."""
    return draw_mmc(m)


def draw_express(m1: dict, m2: dict, pct: float) -> go.Figure:
    """Two-lane express system."""
    fig = _base_fig(2, extra_height=20)

    # ── Express (fast) lane — row 0 at top ────────────────────────────────
    y0 = _row_y(0, 2)
    _add_lane(fig, y0,
              lam=m1["lam"], mu=m1["mu"],
              rho=m1["rho"], Lq=m1["Lq"],
              label="Express Server (fast)", color=BLUE,
              show_wait_legend=True, show_serve_legend=True)

    # Label for split
    fig.add_annotation(x=0.55, y=y0 + 0.02,
        text=f"<b>{pct:.0%}</b>", showarrow=False,
        font=dict(size=11, color=BLUE))

    # ── Regular (slow) lane — row 1 at bottom ─────────────────────────────
    y1 = _row_y(1, 2)
    _add_lane(fig, y1,
              lam=m2["lam"], mu=m2["mu"],
              rho=m2["rho"], Lq=m2["Lq"],
              label="Regular Server (slow)", color=ORANGE,
              show_wait_legend=False, show_serve_legend=False)

    fig.add_annotation(x=0.55, y=y1 + 0.02,
        text=f"<b>{1-pct:.0%}</b>", showarrow=False,
        font=dict(size=11, color=ORANGE))

    # Total λ arrow (left-most)
    fig.add_annotation(
        x=0.1, y=(y0 + y1) / 2 + 0.05,
        text=f"λ<sub>total</sub><br>= {m1['lam']+m2['lam']:.2f}",
        showarrow=False,
        font=dict(size=10, color=GRAY),
        align="center",
    )
    fig.add_shape(type="line", x0=0.15, x1=0.6, y0=y0, y1=y0,
        line_color=BLUE, line_width=2)
    fig.add_shape(type="line", x0=0.15, x1=0.6, y0=y1, y1=y1,
        line_color=ORANGE, line_width=2)
    fig.add_shape(type="line", x0=0.15, x1=0.15, y0=y0, y1=y1,
        line_color="#94a3b8", line_width=1.5)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED METRIC TABLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def metric_table_html(m: dict, color: str = BLUE) -> str:
    rows = [
        ("Arrival rate",             "λ",           fmt(m['lam'],3)),
        ("Service rate (per server)","μ",            fmt(m['mu'],3)),
        ("Number of servers",        "c",            str(m['c'])),
        ("Traffic intensity",        "a = λ / μ",   fmt(m['a'],4)),
        ("Server utilisation",       "ρ = a / c",   fmt(m['rho'],4,pct=True)),
        ("Prob. system empty",       "P₀",          fmt(m['P0'],4,pct=True)),
        ("Prob. customer waits",     "C(c,a)",      fmt(m['Pw'],4,pct=True)),
        ("Avg. customers in queue",  "Lq",          fmt(m['Lq'],4)),
        ("Avg. customers in service","Ls = λ/μ",    fmt(m['Ls'],4)),
        ("Avg. customers in system", "L = Lq + Ls", fmt(m['L'],4)),
        ("Avg. wait in queue",       "Wq",          fmt(m['Wq'],4,time=True)),
        ("Avg. time in system",      "W = Wq + 1/μ",fmt(m['W'],4,time=True)),
    ]
    html = '<table class="metric-tbl"><tr><th>Parameter</th><th>Formula</th><th>Value</th></tr>'
    for param, formula, value in rows:
        hl = (' class="hl"' if param.startswith("Avg.")
              else ' style="color:{};font-weight:600;"'.format(color)
              if "Prob." in param else "")
        html += f'<tr><td>{param}</td><td style="color:#64748b;font-family:monospace;">{formula}</td><td{hl}>{value}</td></tr>'
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

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
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
            rho1_in = st.slider("Per-server utilisation  ρ", 0.01, 0.99, 0.60, 0.01)
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
            color1 = SERVER_COLORS[(c_sel-1) % len(SERVER_COLORS)]

            # KPI row
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.markdown(kpi("Utilisation ρ", f"{m1['rho']:.1%}", color1),
                            unsafe_allow_html=True)
            with kpi_cols[1]:
                st.markdown(kpi("Avg queue  Lq", f"{m1['Lq']:.3f}", ORANGE),
                            unsafe_allow_html=True)
            with kpi_cols[2]:
                st.markdown(kpi("Avg system  L", f"{m1['L']:.3f}", GREEN),
                            unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(kpi("Wait in queue Wq", f"{m1['Wq']:.4f}", RED),
                            unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(metric_table_html(m1, color1), unsafe_allow_html=True)

    # ── Diagram ───────────────────────────────────────────────────────────────
    if m1 is not None:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Queue Diagram</div>', unsafe_allow_html=True)
        st.plotly_chart(draw_mmc(m1), use_container_width=True,
                        config={"displayModeBar": False})

        # ── All-c comparison ──────────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Comparison across c = 1, 2, 3, 4  '
                    '(same λ and μ)</div>', unsafe_allow_html=True)

        comp_rows = []
        for c in [1, 2, 3, 4]:
            mc = mmc(lam1, mu1, c)
            if mc:
                comp_rows.append({
                    "c":   c,
                    "ρ":   f"{mc['rho']:.1%}",
                    "P₀":  f"{mc['P0']:.4f}",
                    "C(c,a)": f"{mc['Pw']:.4f}",
                    "Lq":  f"{mc['Lq']:.4f}",
                    "L":   f"{mc['L']:.4f}",
                    "Wq":  f"{mc['Wq']:.4f}",
                    "W":   f"{mc['W']:.4f}",
                })
            else:
                comp_rows.append({
                    "c": c,
                    "ρ": f"{lam1/(c*mu1):.3f} ⚠",
                    "P₀":"unstable","C(c,a)":"—",
                    "Lq":"—","L":"—","Wq":"—","W":"—",
                })

        df_comp = pd.DataFrame(comp_rows).set_index("c")

        def _hl_row(row):
            if row.name == c_sel:
                return [f"background:{hex_rgba(color1,0.15)}; "
                        f"color:{color1}; font-weight:700;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_comp.style.apply(_hl_row, axis=1),
            use_container_width=True, height=190,
        )
        st.caption(f"▶ Blue row = current selection (c = {c_sel}). "
                   "Adding servers reduces Lq, L, Wq, and W dramatically.")

        # Bar chart comparison
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
        n2   = st.selectbox("Number of servers / queues  (n)", [1,2,3,4], index=1,
                            key="n2")

    # n × M/M/1: each of n queues gets λ/n
    m2_each = mmc(lam2 / n2, mu2, 1)
    # M/M/n: single queue, n servers
    m2_mmn  = mmc(lam2, mu2, n2)

    stable_each = m2_each is not None
    stable_mmn  = m2_mmn  is not None

    if not stable_each:
        st.markdown(
            f'<div class="warn-box">⚠ n×M/M/1 unstable: '
            f'ρ = (λ/n)/μ = {lam2/n2:.2f}/{mu2:.2f} = {lam2/(n2*mu2):.3f} ≥ 1</div>',
            unsafe_allow_html=True,
        )
    elif not stable_mmn:
        st.markdown(
            f'<div class="warn-box">⚠ M/M/{n2} unstable: '
            f'ρ = λ/(n·μ) = {lam2:.2f}/({n2}×{mu2:.2f}) = {lam2/(n2*mu2):.3f} ≥ 1</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── KPI summary ───────────────────────────────────────────────────────
        total_L_nxmm1 = n2 * m2_each["L"]
        total_Lq_nxmm1 = n2 * m2_each["Lq"]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        head_a, head_b = st.columns(2)
        with head_a:
            st.markdown(f'<div class="info-box" style="border-left-color:{ORANGE};">'
                        f'<b style="color:{ORANGE};">{n2} × M/M/1 — {n2} separate queues</b><br>'
                        f'Each queue receives λ/n = {lam2:.2f}/{n2} = {lam2/n2:.3f} '
                        f'arrivals/unit time. Each customer joins one queue and '
                        f'<em>cannot switch</em> even if another queue is shorter.</div>',
                        unsafe_allow_html=True)
        with head_b:
            st.markdown(f'<div class="info-box" style="border-left-color:{BLUE};">'
                        f'<b style="color:{BLUE};">M/M/{n2} — single shared queue</b><br>'
                        f'All {lam2:.2f} arrivals join one queue and the next free server '
                        f'takes the front customer. No lane-switching problem. '
                        f'Always at least as efficient as {n2}×M/M/1.</div>',
                        unsafe_allow_html=True)

        # ── Diagrams ─────────────────────────────────────────────────────────
        diag_a, diag_b = st.columns(2)
        with diag_a:
            st.markdown(f'<div class="sec-hdr">{n2} × M/M/1 system</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(draw_nxmm1(n2, m2_each),
                            use_container_width=True, config={"displayModeBar": False})
        with diag_b:
            st.markdown(f'<div class="sec-hdr">M/M/{n2} system</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(draw_mmn(m2_mmn),
                            use_container_width=True, config={"displayModeBar": False})

        # ── Comparison table ──────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Side-by-side comparison</div>',
                    unsafe_allow_html=True)

        metrics_list = [
            ("Utilisation per server",   "ρ",
             f"{m2_each['rho']:.4f}", f"{m2_mmn['rho']:.4f}"),
            ("Prob. system empty",       "P₀",
             f"{m2_each['P0']:.4f}", f"{m2_mmn['P0']:.4f}"),
            ("Prob. customer waits",     "C(c,a)",
             f"{m2_each['Pw']:.4f}", f"{m2_mmn['Pw']:.4f}"),
            ("Avg. customers in queue (each)",  "Lq",
             f"{m2_each['Lq']:.4f}", f"{m2_mmn['Lq']:.4f}"),
            ("Avg. customers in system (each)", "L",
             f"{m2_each['L']:.4f}", f"{m2_mmn['L']:.4f}"),
            ("★ Total customers in system",     "L_total",
             f"{total_L_nxmm1:.4f}", f"{m2_mmn['L']:.4f}"),
            ("★ Total customers in queue",      "Lq_total",
             f"{total_Lq_nxmm1:.4f}", f"{m2_mmn['Lq']:.4f}"),
            ("Avg. wait in queue",       "Wq",
             f"{m2_each['Wq']:.4f}", f"{m2_mmn['Wq']:.4f}"),
            ("Avg. time in system",      "W",
             f"{m2_each['W']:.4f}", f"{m2_mmn['W']:.4f}"),
        ]

        tbl  = (f'<table class="cmp-tbl">'
                f'<tr><th>Metric</th><th>Formula</th>'
                f'<th style="color:{ORANGE};">{n2} × M/M/1</th>'
                f'<th style="color:{BLUE};">M/M/{n2}</th>'
                f'<th>Better?</th></tr>')

        for param, formula, v_nxmm1, v_mmn in metrics_list:
            try:
                a_val = float(v_nxmm1); b_val = float(v_mmn)
                if b_val < a_val:
                    cls_a = " class='worse'"; cls_b = " class='better'"
                    better = f'<span style="color:{BLUE}">M/M/{n2} ✓</span>'
                elif a_val < b_val:
                    cls_a = " class='better'"; cls_b = " class='worse'"
                    better = f'<span style="color:{ORANGE}">{n2}×M/M/1 ✓</span>'
                else:
                    cls_a = cls_b = ""; better = "Equal"
            except Exception:
                cls_a = cls_b = ""; better = "—"

            star = " style='font-weight:700;'" if param.startswith("★") else ""
            tbl += (f'<tr><td{star}>{param}</td>'
                    f'<td style="color:#64748b;font-family:monospace;">{formula}</td>'
                    f'<td{cls_a}>{v_nxmm1}</td>'
                    f'<td{cls_b}>{v_mmn}</td>'
                    f'<td style="font-size:12px;">{better}</td></tr>')
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

        # Improvement bar chart
        improvement_Lq = (total_Lq_nxmm1 - m2_mmn["Lq"]) / total_Lq_nxmm1 * 100
        improvement_Wq = (m2_each["Wq"] - m2_mmn["Wq"]) / m2_each["Wq"] * 100

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box" style="border-left-color:{GREEN};">'
            f'<b style="color:{GREEN};">Pooling effect:</b> '
            f'The M/M/{n2} single-queue system reduces total queue length (Lq) by '
            f'<b>{improvement_Lq:.1f}%</b> and waiting time (Wq) by '
            f'<b>{improvement_Wq:.1f}%</b> compared to {n2} separate M/M/1 queues '
            f'with the same total capacity.</div>',
            unsafe_allow_html=True,
        )

        fig_cmp = go.Figure()
        labels = [f"{n2}×M/M/1", f"M/M/{n2}"]
        fig_cmp.add_trace(go.Bar(x=labels,
            y=[total_Lq_nxmm1, m2_mmn["Lq"]],
            name="Total Lq", marker_color=[ORANGE, BLUE], opacity=0.85,
            text=[f"{total_Lq_nxmm1:.3f}", f"{m2_mmn['Lq']:.3f}"],
            textposition="outside"))
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
# ║  TAB 3 — Express Lines (2 × M/M/1, split λ)                             ║
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

    err_f = m3f is None
    err_s = m3s is None

    if err_f or err_s:
        msgs = []
        if err_f:
            msgs.append(f"Express lane unstable: ρ₁ = {lam3_fast:.2f}/{mu3_fast:.2f} = "
                        f"{lam3_fast/mu3_fast:.3f} ≥ 1. Increase μ₁ or reduce λ.")
        if err_s:
            msgs.append(f"Regular lane unstable: ρ₂ = {lam3_slow:.2f}/{mu3_slow:.2f} = "
                        f"{lam3_slow/mu3_slow:.3f} ≥ 1. Increase μ₂ or reduce λ.")
        for msg in msgs:
            st.markdown(f'<div class="warn-box">⚠ {msg}</div>',
                        unsafe_allow_html=True)
    else:
        # ── System-level aggregates ───────────────────────────────────────────
        L_total   = m3f["L"]  + m3s["L"]
        Lq_total  = m3f["Lq"] + m3s["Lq"]
        Wq_avg    = (lam3_fast*m3f["Wq"] + lam3_slow*m3s["Wq"]) / lam3_total
        W_avg     = (lam3_fast*m3f["W"]  + lam3_slow*m3s["W"])  / lam3_total

        # KPIs
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

        # ── Diagram ───────────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">System Diagram</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(draw_express(m3f, m3s, pct3),
                        use_container_width=True, config={"displayModeBar": False})

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

        # ── System summary table ──────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">System-Level Summary</div>',
                    unsafe_allow_html=True)

        sys_rows = [
            ("Total arrival rate",      "λ",          f"{lam3_total:.3f}"),
            ("Express lane arrivals",   "λ₁ = p·λ",   f"{lam3_fast:.3f} ({pct3:.0%})"),
            ("Regular lane arrivals",   "λ₂ = (1−p)·λ",f"{lam3_slow:.3f} ({1-pct3:.0%})"),
            ("Express server rate",     "μ₁",          f"{mu3_fast:.3f}"),
            ("Regular server rate",     "μ₂",          f"{mu3_slow:.3f}"),
            ("Express utilisation",     "ρ₁ = λ₁/μ₁",  f"{m3f['rho']:.4f} = {m3f['rho']:.1%}"),
            ("Regular utilisation",     "ρ₂ = λ₂/μ₂",  f"{m3s['rho']:.4f} = {m3s['rho']:.1%}"),
            ("Total Lq (both lanes)",   "Lq₁ + Lq₂",   f"{Lq_total:.4f}"),
            ("Total L  (both lanes)",   "L₁ + L₂",     f"{L_total:.4f}"),
            ("Weighted avg Wq",         "Σλᵢ·Wqᵢ / λ", f"{Wq_avg:.4f}"),
            ("Weighted avg W",          "Σλᵢ·Wᵢ / λ",  f"{W_avg:.4f}"),
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

        # ── Sensitivity: vary split % ─────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Sensitivity — Wq vs. Express Split %</div>',
                    unsafe_allow_html=True)

        splits   = np.linspace(0.05, 0.95, 80)
        wq_curve = []
        for p in splits:
            mf_ = mmc(p * lam3_total, mu3_fast, 1)
            ms_ = mmc((1-p) * lam3_total, mu3_slow, 1)
            if mf_ and ms_:
                wq_curve.append((p*lam3_total*mf_["Wq"] +
                                 (1-p)*lam3_total*ms_["Wq"]) / lam3_total)
            else:
                wq_curve.append(None)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=splits * 100, y=wq_curve, mode="lines",
            line=dict(color=BLUE, width=2.5),
            name="Weighted Wq",
            hovertemplate="Split: %{x:.0f}%<br>Wq: %{y:.4f}<extra></extra>",
        ))
        fig_sens.add_vline(x=pct3 * 100,
            line_color=RED, line_width=1.5, line_dash="dash",
            annotation_text=f"  Current: {pct3:.0%}",
            annotation_font=dict(color=RED, size=11))
        fig_sens.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            height=270,
            margin=dict(t=14, b=48, l=60, r=20),
            xaxis=dict(title="% routed to express lane",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            yaxis=dict(title="Weighted average Wq",
                       gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            showlegend=False,
        )
        st.plotly_chart(fig_sens, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("The curve shows how the weighted average waiting time changes "
                   "as the split % varies. The red dashed line marks the current setting. "
                   "Gaps in the curve indicate unstable configurations (ρ ≥ 1).")
