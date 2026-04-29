import sys, os, base64
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from agent import GreenhouseAgent

st.set_page_config(page_title="Smart Green House Artificial Intelligence", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

def img64(filename):
    path = os.path.join(os.path.dirname(__file__), "images", filename)
    if not os.path.exists(path): return ""
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

IMGS = {k: img64(v) for k, v in {
    "header":"header.jpg","plants":"plants.jpg","farm":"farm.jpg",
    "leaves":"leaves.jpg","greenhouse":"greenhouse.jpg","water":"water.jpg",
    "sunlight":"sunlight.jpg","frost":"frost.jpg","soil":"soil.jpg",
    "rain":"rain.jpg","crops":"crops.jpg","nature2":"nature2.jpg",
}.items()}

def bg(key, overlay="rgba(255,255,255,0.0)"):
    b64 = IMGS.get(key, "")
    if b64:
        return f"background:linear-gradient({overlay},{overlay}),url(data:image/jpeg;base64,{b64});background-size:cover;background-position:center;"
    return "background:#f0f7ed;"

# ══════════════════════════════════════════════════════════════════════════════
# SOFT GARDEN — Light & Airy Design System
# bg: #edf5ea   surface: #ffffff   sidebar: #fdf6ee
# primary: #2d7a4f  accent: #4a9e6a  warm: #c8714a  cool: #4a9ab5
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background: #edf5ea !important;
}

/* ── Top bar: invisible / seamless ── */
header[data-testid="stHeader"] {
    background: #edf5ea !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton            { display: none !important; }
#MainMenu                  { visibility: hidden !important; }
footer                     { display: none !important; }

/* ── Keep sidebar collapse/expand button visible & clickable ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"] {
    opacity: 1 !important;
    pointer-events: auto !important;
    visibility: visible !important;
    z-index: 999999 !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    color: #2d7a4f !important;
    fill: #2d7a4f !important;
}

/* ── Sidebar: warm cream / linen ── */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #fdf6ee 0%, #faf0e4 100%) !important;
    border-right: 1px solid #e8d8c4 !important;
}

/* ── Global ── */
div[data-testid="stMarkdownContainer"] * { cursor: default !important; user-select: none !important; }

/* ── Metrics ── */
div[data-testid="stMetricValue"] {
    font-size: 1.4rem !important; font-weight: 700 !important; color: #1a3a24 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: .67rem !important; letter-spacing: .8px;
    text-transform: uppercase; color: #6a9a78 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f0f7ed; border-radius: 12px; padding: 4px; gap: 3px;
    border: 1px solid #d4e8cc;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-size: .79rem; font-weight: 600;
    color: #5a8a68; padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #2d7a4f !important;
    box-shadow: 0 2px 8px rgba(45,122,79,0.15) !important;
}

/* ── Buttons (sidebar) ── */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: .81rem !important; background: #ffffff !important;
    color: #3d6b4a !important; border: 1px solid #c8dfc4 !important;
    transition: all .2s !important;
    box-shadow: 0 1px 4px rgba(45,122,79,0.08) !important;
}
.stButton > button:hover {
    background: #f0f7ed !important; border-color: #4a9e6a !important;
    color: #2d7a4f !important; box-shadow: 0 2px 8px rgba(45,122,79,0.14) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2d7a4f, #1e6040) !important;
    color: #ffffff !important; border: none !important;
    box-shadow: 0 4px 16px rgba(45,122,79,0.30) !important;
}

/* ── DataFrame ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden;
    border: 1px solid #d4e8cc !important;
    box-shadow: 0 2px 12px rgba(45,122,79,0.08) !important;
}

/* ── Spinner / success ── */
.stSpinner > div { border-top-color: #2d7a4f !important; }
.stSuccess { background: #f0faf4 !important; color: #2d7a4f !important; border: 1px solid #c8dfc4 !important; }
</style>
""", unsafe_allow_html=True)

# ── Labels & Maps ─────────────────────────────────────────────────────────────
FACT_LABELS = {
    "temp_high":"High Temperature","temp_low":"Low Temperature","temp_normal":"Normal Temperature",
    "humidity_low":"Low Humidity","humidity_high":"High Humidity","humidity_normal":"Normal Humidity",
    "soil_dry":"Dry Soil","soil_wet":"Wet Soil","soil_normal":"Normal Soil Moisture",
    "light_low":"Low Light","co2_high":"High CO2","co2_low":"Low CO2",
}
ACTION_LABELS = {
    "ACTIVATE_IRRIGATION":"Activate Irrigation","DEACTIVATE_IRRIGATION":"Deactivate Irrigation",
    "ACTIVATE_HEATING":"Activate Heating","ACTIVATE_COOLING":"Activate Cooling",
    "OPEN_VENTILATION":"Open Ventilation","ACTIVATE_ARTIFICIAL_LIGHT":"Activate Artificial Light",
    "MAINTAIN_STATE":"Maintain Current State","MONITOR":"Continue Monitoring",
}
ACTION_COLORS = {
    "ACTIVATE_IRRIGATION":       "#4a9ab5",
    "DEACTIVATE_IRRIGATION":     "#2d7a4f",
    "ACTIVATE_HEATING":          "#c8714a",
    "ACTIVATE_COOLING":          "#4a9ab5",
    "OPEN_VENTILATION":          "#6aaa40",
    "ACTIVATE_ARTIFICIAL_LIGHT": "#c8a040",
    "MAINTAIN_STATE":            "#2d7a4f",
    "MONITOR":                   "#6a9a78",
}
ACTION_ICONS = {
    "ACTIVATE_IRRIGATION":"💧","DEACTIVATE_IRRIGATION":"🚫","ACTIVATE_HEATING":"🔥",
    "ACTIVATE_COOLING":"❄️","OPEN_VENTILATION":"🌬️","ACTIVATE_ARTIFICIAL_LIGHT":"💡",
    "MAINTAIN_STATE":"🌿","MONITOR":"👁️",
}

CBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(240,247,237,1)",
    font=dict(color="#5a8a68", size=11, family="DM Sans"),
    height=300,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#5a8a68")),
    xaxis=dict(gridcolor="rgba(45,122,79,0.10)", linecolor="#d4e8cc", showline=True),
    yaxis=dict(gridcolor="rgba(45,122,79,0.10)", linecolor="#d4e8cc", showline=True),
)


def init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = GreenhouseAgent()
        st.session_state.step_count = 0
        st.session_state.readings_df = pd.DataFrame(columns=[
            "step","temperature","humidity","soil_moisture","light_level","co2_level"])
        st.session_state.last_result = None
        st.session_state.optimization_result = None

def run_step():
    r = agent.step()
    st.session_state.step_count += 1
    st.session_state.last_result = r
    rd = r["reading"]
    st.session_state.readings_df = pd.concat([st.session_state.readings_df,
        pd.DataFrame([{"step":st.session_state.step_count,"temperature":rd.temperature,
            "humidity":rd.humidity,"soil_moisture":rd.soil_moisture,
            "light_level":rd.light_level,"co2_level":rd.co2_level}])],ignore_index=True)

def make_gauge(value, lo, hi, opt_lo, opt_hi, unit, title, fmt=".1f"):
    color = "#2d7a4f" if opt_lo<=value<=opt_hi else ("#c8a040" if value<opt_lo else "#c8714a")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":f" {unit}", "valueformat":fmt,
                "font":{"size":22, "color":"#1a3a24", "family":"DM Sans"}},
        title={"text":title, "font":{"size":10, "color":"#6a9a78", "family":"DM Sans"}},
        gauge={
            "axis":{"range":[lo,hi],"tickfont":{"size":8,"color":"#a8c8b0"},"nticks":5,"tickcolor":"#d4e8cc"},
            "bar":{"color":color,"thickness":.26},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[
                {"range":[lo, opt_lo],  "color":"rgba(200,160,64,.08)"},
                {"range":[opt_lo,opt_hi],"color":"rgba(45,122,79,.09)"},
                {"range":[opt_hi, hi],  "color":"rgba(200,113,74,.08)"}],
            "threshold":{"line":{"color":color,"width":2.5},"thickness":.8,"value":value}},
    ))
    fig.update_layout(height=185, margin=dict(l=14,r=14,t=34,b=6),
        paper_bgcolor="rgba(255,255,255,0.95)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def sec(label, color="#2d7a4f"):
    st.markdown(
        f'<div style="font-size:.69rem;font-weight:700;letter-spacing:1.4px;'
        f'text-transform:uppercase;color:{color};border-left:3px solid {color};'
        f'padding-left:10px;margin:26px 0 14px 0;">{label}</div>',
        unsafe_allow_html=True)

def card(content, padding="18px", extra=""):
    return (f'<div style="background:#ffffff;border:1px solid #ddeedd;border-radius:16px;'
            f'padding:{padding};box-shadow:0 2px 16px rgba(45,122,79,0.08);{extra}">{content}</div>')


init_state()
agent: GreenhouseAgent = st.session_state.agent

# ══════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
header_bg = bg("header", "rgba(10,30,14,0.52)")
st.markdown(f"""
<div style="{header_bg}border-radius:22px;padding:38px 50px;margin-bottom:28px;
            border:1px solid #c8e0c4;
            box-shadow:0 4px 32px rgba(45,122,79,0.12);overflow:hidden;">
    <div style="display:flex;align-items:center;gap:22px;margin-bottom:14px;">
        <div style="background:rgba(255,255,255,0.18);backdrop-filter:blur(8px);
                    border:1px solid rgba(255,255,255,0.30);border-radius:18px;
                    padding:12px 16px;font-size:2.4rem;line-height:1;">🌿</div>
        <div>
            <div style="font-size:1.85rem;font-weight:800;color:#ffffff;letter-spacing:-.3px;
                        text-shadow:0 2px 16px rgba(0,0,0,.4);">
                Smart Green House Artificial Intelligence</div>
            <div style="font-size:.86rem;color:rgba(255,255,255,.80);margin-top:6px;font-weight:400;">
                Autonomous rational agent for real-time greenhouse environment management
            </div>
        </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px;">
        {''.join(
            f'<span style="background:rgba(255,255,255,.18);backdrop-filter:blur(6px);'
            f'color:#ffffff;border:1px solid rgba(255,255,255,.28);border-radius:20px;'
            f'padding:4px 14px;font-size:.72rem;font-weight:600;">{t}</span>'
            for t in ["🧠 Propositional Logic","📐 Linear Algebra & PCA",
                       "📊 Probability Theory","🧬 Genetic Algorithm","📡 Real-Time Monitoring"])}
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    running = st.session_state.step_count > 0

    if running:
        box_bg   = "linear-gradient(135deg,#2d7a4f,#1e6040)"
        box_bdr  = "#4a9e6a"
        box_shad = "0 6px 24px rgba(45,122,79,0.35)"
        dot_c    = "#a0e8b8"
        lbl_c    = "rgba(255,255,255,.75)"
        txt_c    = "#ffffff"
        sub_c    = "rgba(255,255,255,.70)"
        s_txt    = "🟢 Simulating"
        s_sub    = f"Step {st.session_state.step_count} completed"
    else:
        box_bg   = "linear-gradient(135deg,#4a9e6a,#2d7a4f)"
        box_bdr  = "#86c99a"
        box_shad = "0 6px 28px rgba(45,122,79,0.30)"
        dot_c    = "#d4f0e0"
        lbl_c    = "rgba(255,255,255,.75)"
        txt_c    = "#ffffff"
        sub_c    = "rgba(255,255,255,.70)"
        s_txt    = "⏸ Standby"
        s_sub    = "Press ▶ Step to begin"

    st.markdown(f"""
    <div style="background:{box_bg};border:1.5px solid {box_bdr};
                border-radius:18px;padding:26px 18px 22px;margin-bottom:22px;
                box-shadow:{box_shad};text-align:center;">
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:10px;">
            <div style="width:9px;height:9px;border-radius:50%;background:{dot_c};
                        box-shadow:0 0 8px {dot_c};"></div>
            <span style="color:{lbl_c};font-size:.68rem;font-weight:700;
                         letter-spacing:1.5px;text-transform:uppercase;">Agent Status</span>
        </div>
        <div style="color:{txt_c};font-size:1.75rem;font-weight:800;
                    text-shadow:0 2px 10px rgba(0,0,0,.18);">{s_txt}</div>
        <div style="color:{sub_c};font-size:.78rem;margin-top:8px;font-weight:500;">{s_sub}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p style="color:#a07850;font-size:.67rem;font-weight:700;letter-spacing:1.2px;'
                'text-transform:uppercase;margin:0 0 8px 0;">Simulation Controls</p>',
                unsafe_allow_html=True)
    ca, cb   = st.columns(2)
    step_btn     = ca.button("▶  Step",          use_container_width=True, type="primary")
    run10_btn    = cb.button("⏩  × 10",          use_container_width=True)
    run50_btn    = st.button("⏭  Run 50 Steps",   use_container_width=True)
    optimize_btn = st.button("🧬  Optimize (GA)", use_container_width=True)

    st.markdown('<p style="color:#a07850;font-size:.67rem;font-weight:700;letter-spacing:1.2px;'
                'text-transform:uppercase;margin:18px 0 8px 0;">Manual Overrides</p>',
                unsafe_allow_html=True)
    irrigate_btn = st.button("💧  Force Irrigate",   use_container_width=True)
    heat_btn     = st.button("🔥  Force Heat +2°C",  use_container_width=True)
    cool_btn     = st.button("❄️  Force Cool −2°C",  use_container_width=True)

    if st.session_state.last_result:
        mm = st.session_state.last_result["math_metrics"]
        st.markdown('<p style="color:#a07850;font-size:.67rem;font-weight:700;letter-spacing:1.2px;'
                    'text-transform:uppercase;margin:18px 0 8px 0;">Live Summary</p>',
                    unsafe_allow_html=True)
        st.metric("Total Steps",        st.session_state.step_count)
        st.metric("Growth Probability", f"{mm['growth_probability']:.1%}")
        st.metric("Anomaly Score",      f"{mm['anomaly_score']:.3f}")
        st.metric("Expected Yield",     f"{mm['expected_yield_30d']:.2f} kg / 30d")

    st.markdown("""
    <div style="margin-top:24px;">
        <a href="https://share.streamlit.io" target="_blank"
           style="display:block;background:rgba(45,122,79,0.08);
                  border:1px solid #c8dfc4;border-radius:10px;
                  padding:10px 14px;text-decoration:none;text-align:center;">
            <span style="color:#2d7a4f;font-size:.75rem;font-weight:700;letter-spacing:.8px;">
                🚀 Deploy App
            </span>
        </a>
    </div>""", unsafe_allow_html=True)

if step_btn:  run_step()
if run10_btn: [run_step() for _ in range(10)]
if run50_btn: [run_step() for _ in range(50)]
if optimize_btn:
    with st.spinner("Growing the optimal schedule… 🌱"):
        st.session_state.optimization_result = agent.optimize_schedule()
if irrigate_btn: agent.sensor.irrigate(15.0); st.sidebar.success("Irrigation +15%")
if heat_btn:     agent.sensor.heat(2.0);      st.sidebar.success("Heating +2°C")
if cool_btn:     agent.sensor.cool(2.0);      st.sidebar.success("Cooling −2°C")

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.last_result is None:
    w1, w2, w3 = st.columns(3, gap="large")

    for col, img_key, accent, title_txt, content in [
        (w1, "nature2", "#2d7a4f", "🌿 Agent Cycle",
         "".join(
             f'<div style="display:flex;gap:12px;margin-bottom:10px;align-items:flex-start;">'
             f'<div style="background:rgba(45,122,79,.12);color:#2d7a4f;border-radius:50%;'
             f'width:22px;height:22px;min-width:22px;display:flex;align-items:center;justify-content:center;'
             f'font-size:.66rem;font-weight:700;margin-top:1px;">{i}</div>'
             f'<div style="color:#3d6b4a;font-size:.8rem;line-height:1.6;">{t}</div></div>'
             for i,t in [
                 (1,'<strong style="color:#1a3a24;">Perceive</strong> — Read all 5 sensor channels'),
                 (2,'<strong style="color:#1a3a24;">Think</strong> — Apply 8 rules via Modus Ponens'),
                 (3,'<strong style="color:#1a3a24;">Evaluate</strong> — Anomaly score, growth probability'),
                 (4,'<strong style="color:#1a3a24;">Optimize</strong> — GA plans irrigation schedule'),
                 (5,'<strong style="color:#1a3a24;">Act</strong> — Drive actuators from conclusions')])),
        (w2, "water", "#4a9ab5", "🧠 Logic Rules",
         "".join(
             f'<div style="display:flex;align-items:center;padding:6px 10px;margin-bottom:5px;'
             f'background:#f5faf3;border-radius:8px;border-left:3px solid {c};">'
             f'<span style="color:#a8c8b0;font-size:.64rem;font-weight:700;width:22px;">{r}</span>'
             f'<span style="color:#5a8a68;font-size:.71rem;flex:1;margin:0 6px;">{cond}</span>'
             f'<span style="color:{c};font-size:.69rem;font-weight:700;white-space:nowrap;">{act}</span></div>'
             for r,cond,act,c in [
                 ("R1","High Temp ∧ Low Humidity","Cooling","#4a9ab5"),
                 ("R2","High Temperature","Ventilation","#6aaa40"),
                 ("R3","Low Temperature","Heating","#c8714a"),
                 ("R4","Dry Soil","Irrigation","#2d7a4f"),
                 ("R5","Wet Soil","Stop Irrigation","#6a9a78"),
                 ("R6","Low Light","Artificial Light","#c8a040"),
                 ("R7","High CO2 ∨ High Humidity","Ventilation","#6aaa40"),
                 ("R8","All Normal","Maintain State","#2d7a4f")])),
        (w3, "crops", "#c8a040", "🧬 Genetic Algorithm",
         "".join(
             f'<div style="display:flex;gap:10px;margin-bottom:10px;">'
             f'<span style="font-size:.88rem;width:20px;min-width:20px;">{ic}</span>'
             f'<div style="color:#5a8a68;font-size:.8rem;line-height:1.5;">'
             f'<strong style="color:#1a3a24;">{lb}</strong> — {ds}</div></div>'
             for ic,lb,ds in [
                 ("🧬","Chromosome","24 binary genes, one per hour"),
                 ("🎯","Fitness","health_score − water_cost_penalty"),
                 ("🏆","Selection","Tournament selection  (k = 3)"),
                 ("✂️","Crossover","Single-point  (rate = 0.8)"),
                 ("🎲","Mutation","Bit-flip  (rate = 0.05)"),
                 ("👑","Elitism","Best individual always preserved"),
                 ("🔁","Generations","60 iterations per run")])),
    ]:
        img_strip = (
            f'<div style="{bg(img_key,"rgba(8,24,12,0.30)")}height:108px;border-radius:12px;'
            f'margin-bottom:16px;display:flex;align-items:flex-end;padding:10px 14px;">'
            f'<span style="color:#ffffff;font-size:.67rem;font-weight:700;letter-spacing:1.2px;'
            f'text-transform:uppercase;text-shadow:0 1px 8px rgba(0,0,0,.7);">{title_txt}</span></div>'
        )
        col.markdown(
            card(img_strip + content, padding="16px", extra="height:380px;overflow:hidden;cursor:default;user-select:none;"),
            unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #c8e0c4;border-radius:16px;
                margin-top:28px;padding:32px 40px;
                box-shadow:0 2px 16px rgba(45,122,79,0.08);text-align:center;">
        <div style="font-size:1.9rem;margin-bottom:10px;">🌱</div>
        <div style="color:#2d7a4f;font-size:1rem;font-weight:700;">
            Press <strong>Step</strong> in the sidebar to start the simulation
        </div>
        <div style="color:#a8c8b0;font-size:.78rem;margin-top:7px;">
            Each step runs one full agent cycle: Perceive → Think → Optimize → Act
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
else:
    result      = st.session_state.last_result
    reading     = result["reading"]
    facts       = result["facts"]
    conclusions = result["conclusions"]
    mm          = result["math_metrics"]
    actuators   = result["actuators"]

    # — Gauges ─────────────────────────────────────────────────────────────────
    sec("🌡️  Live Sensor Readings", "#2d7a4f")
    g1,g2,g3,g4,g5 = st.columns(5, gap="small")
    g1.plotly_chart(make_gauge(reading.temperature,   0, 45, 15, 28,   "°C","Temperature"),       use_container_width=True)
    g2.plotly_chart(make_gauge(reading.humidity,      0,100, 50, 85,    "%","Humidity"),           use_container_width=True)
    g3.plotly_chart(make_gauge(reading.soil_moisture, 0,100, 45, 75,    "%","Soil Moisture"),      use_container_width=True)
    g4.plotly_chart(make_gauge(reading.light_level,   0,12000,3000,9000,"lux","Light Level",".0f"),use_container_width=True)
    g5.plotly_chart(make_gauge(reading.co2_level,   400,2000,600,1200,  "ppm","CO2 Level",".0f"), use_container_width=True)

    # — Banner strip ───────────────────────────────────────────────────────────
    rain_bg = bg("rain", "rgba(8,24,12,0.45)")
    st.markdown(f"""
    <div style="{rain_bg}height:66px;border-radius:14px;margin:10px 0 4px 0;
                border:1px solid #c8e0c4;display:flex;align-items:center;padding:0 26px;
                box-shadow:0 2px 12px rgba(45,122,79,0.10);">
        <span style="color:#ffffff;font-size:.76rem;font-weight:600;
                     text-shadow:0 1px 8px rgba(0,0,0,.6);">
            🌿 &nbsp; Greenhouse Environment Control System &nbsp;·&nbsp;
            Rational Agent Active &nbsp;·&nbsp; Step {st.session_state.step_count}
        </span>
    </div>""", unsafe_allow_html=True)

    # — Actuators ──────────────────────────────────────────────────────────────
    sec("⚙️  Actuator Status", "#4a9ab5")
    a1,a2,a3,a4,a5 = st.columns(5, gap="small")
    ACT_ON_COLORS = {
        "irrigation":"#4a9ab5","heating":"#c8714a","cooling":"#4a9ab5",
        "ventilation":"#6aaa40","artificial_light":"#c8a040"}
    for col,icon,key,label in [
        (a1,"💧","irrigation","Irrigation"),(a2,"🔥","heating","Heating"),
        (a3,"❄️","cooling","Cooling"),(a4,"🌬️","ventilation","Ventilation"),
        (a5,"💡","artificial_light","Artificial Light")]:
        on = actuators[key]
        ac = ACT_ON_COLORS.get(key,"#2d7a4f")
        col.markdown(f"""
        <div style="background:#ffffff;
                    border:{'2px solid '+ac if on else '1px solid #ddeedd'};
                    border-radius:14px;padding:18px 8px;text-align:center;
                    box-shadow:{'0 4px 18px '+ac+'28' if on else '0 1px 6px rgba(45,122,79,0.06)'};
                    cursor:default;user-select:none;">
            <div style="font-size:1.5rem;margin-bottom:7px;">{icon}</div>
            <div style="color:{'#1a3a24' if on else '#a8c8b0'};font-size:.65rem;font-weight:700;
                        letter-spacing:.9px;text-transform:uppercase;margin-bottom:7px;">{label}</div>
            <div style="background:{ac+'18' if on else '#f0f7ed'};
                        color:{ac if on else '#c8dfc4'};border-radius:20px;
                        padding:3px 10px;font-size:.67rem;font-weight:700;display:inline-block;">
                {'● ACTIVE' if on else '○  IDLE'}</div>
        </div>""", unsafe_allow_html=True)

    # — Logic + Math ───────────────────────────────────────────────────────────
    sec("🧠  Logic Engine  ·  📐  Mathematical Models", "#7c6bb0")
    lc, mc = st.columns([1,1], gap="large")

    with lc:
        active_f   = [k for k,v in facts.items() if v]
        inactive_f = [k for k,v in facts.items() if not v]
        chips = (
            "".join(
                f'<span style="display:inline-block;background:rgba(45,122,79,0.10);color:#2d7a4f;'
                f'border:1px solid rgba(45,122,79,0.20);border-radius:20px;padding:4px 12px;'
                f'font-size:.71rem;font-weight:600;margin:3px;">✓ {FACT_LABELS.get(f,f)}</span>'
                for f in active_f) +
            "".join(
                f'<span style="display:inline-block;background:#f5faf3;color:#c8dfc4;'
                f'border:1px solid #e8f0e4;border-radius:20px;padding:4px 12px;'
                f'font-size:.71rem;margin:3px;">{FACT_LABELS.get(f,f)}</span>'
                for f in inactive_f))
        st.markdown(card(
            f'<div style="color:#a8c8b0;font-size:.64rem;font-weight:700;letter-spacing:1.3px;'
            f'text-transform:uppercase;margin-bottom:10px;">Active Propositions</div>'
            f'<div style="line-height:2.2;">{chips}</div>',
            padding="18px 18px 14px"), unsafe_allow_html=True)

        st.markdown('<div style="color:#a8c8b0;font-size:.64rem;font-weight:700;letter-spacing:1.3px;'
                    'text-transform:uppercase;margin:14px 0 9px 0;">Inference Results — Modus Ponens</div>',
                    unsafe_allow_html=True)
        for c in conclusions:
            color = ACTION_COLORS.get(c.action, "#2d7a4f")
            icon  = ACTION_ICONS.get(c.action, "🌿")
            label = ACTION_LABELS.get(c.action, c.action.replace("_"," ").title())
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid {color}28;border-left:4px solid {color};
                        border-radius:12px;padding:12px 16px;margin-bottom:8px;
                        box-shadow:0 1px 8px rgba(0,0,0,0.04);cursor:default;user-select:none;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    <span style="font-size:.9rem;">{icon}</span>
                    <span style="color:{color};font-size:.82rem;font-weight:700;">{label}</span>
                </div>
                <div style="color:#a8c8b0;font-size:.66rem;font-family:monospace;margin-bottom:3px;">{c.rule}</div>
                <div style="color:#5a8a68;font-size:.75rem;">{c.reason}</div>
            </div>""", unsafe_allow_html=True)

    with mc:
        gp       = mm["growth_probability"]
        gp_color = "#2d7a4f" if gp>.6 else ("#c8a040" if gp>.3 else "#c8714a")
        gp_rgb   = "45,122,79" if gp>.6 else ("200,160,64" if gp>.3 else "200,113,74")
        gp_label = "Excellent" if gp>.7 else ("Good" if gp>.5 else ("Fair" if gp>.3 else "Poor"))
        soil_bg  = bg("soil", "rgba(8,24,12,0.42)")
        st.markdown(f"""
        <div style="{soil_bg}border:1px solid #c8e0c4;border-radius:16px;
                    padding:22px;margin-bottom:16px;
                    box-shadow:0 2px 16px rgba(45,122,79,0.10);cursor:default;user-select:none;">
            <div style="color:rgba(255,255,255,.65);font-size:.64rem;font-weight:700;
                        letter-spacing:1.3px;text-transform:uppercase;margin-bottom:12px;">
                🌱 Growth Probability</div>
            <div style="display:flex;align-items:flex-end;gap:12px;margin-bottom:12px;">
                <div style="font-size:3rem;font-weight:800;color:#ffffff;line-height:1;
                            text-shadow:0 2px 12px rgba(0,0,0,.3);">{gp:.1%}</div>
                <div style="margin-bottom:6px;">
                    <span style="background:rgba(255,255,255,.20);color:#ffffff;border-radius:20px;
                                 padding:3px 12px;font-size:.72rem;font-weight:700;">{gp_label}</span>
                </div>
            </div>
            <div style="background:rgba(0,0,0,.20);border-radius:20px;height:8px;overflow:hidden;">
                <div style="width:{gp*100:.1f}%;height:100%;
                            background:rgba(255,255,255,.75);border-radius:20px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        anomaly = mm["anomaly_score"]
        a_color = "#2d7a4f" if anomaly<2 else ("#c8a040" if anomaly<4 else "#c8714a")
        a_label = "Normal" if anomaly<2 else ("Caution" if anomaly<4 else "Anomaly Detected")
        stats   = agent.math.get_statistics()
        rows    = [
            ("Expected Yield (30 days)", f"{mm['expected_yield_30d']:.2f} kg", "#1a3a24"),
            ("L2 Norm of State Vector",  f"{mm['l2_norm']:.4f}",               "#1a3a24"),
            ("Anomaly Score",            f"{anomaly:.3f}  —  {a_label}",       a_color),
        ]
        if stats and "top_eigenvalue" in stats:
            rows += [
                ("Sensor Matrix Shape",    str(stats["sensor_matrix_shape"]),          "#1a3a24"),
                ("Top Eigenvalue  λ₁",     str(stats["top_eigenvalue"]),               "#4a9ab5"),
                ("PCA Explained Variance", f"{stats['explained_variance_ratio']:.1%}", "#7c6bb0"),
            ]
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:8px 0;border-bottom:1px solid #eef5ec;">'
            f'<span style="color:#a8c8b0;font-size:.73rem;">{k}</span>'
            f'<span style="color:{vc};font-size:.75rem;font-weight:600;font-family:monospace;">{v}</span></div>'
            for k,v,vc in rows)
        st.markdown(card(
            f'<div style="color:#a8c8b0;font-size:.64rem;font-weight:700;letter-spacing:1.3px;'
            f'text-transform:uppercase;margin-bottom:11px;">Linear Algebra &amp; Probability</div>'
            + rows_html), unsafe_allow_html=True)

    # — Charts ─────────────────────────────────────────────────────────────────
    df = st.session_state.readings_df
    if len(df) > 1:
        sec("📈  Sensor History", "#2d7a4f")
        t1, t2, t3 = st.tabs(["🌡️  Temperature & Humidity","🌱  Soil & Light","💨  CO2"])
        with t1:
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=df["step"],y=df["temperature"],name="Temperature (°C)",
                line=dict(color="#c8714a",width=2.5),fill="tozeroy",fillcolor="rgba(200,113,74,.08)"),secondary_y=False)
            fig.add_trace(go.Scatter(x=df["step"],y=df["humidity"],name="Humidity (%)",
                line=dict(color="#4a9ab5",width=2.5)),secondary_y=True)
            fig.add_hline(y=28,line_dash="dot",line_color="rgba(200,113,74,.4)",secondary_y=False,
                annotation_text="High Limit",annotation_font=dict(color="#c8714a",size=10))
            fig.add_hline(y=15,line_dash="dot",line_color="rgba(74,154,181,.4)",secondary_y=False,
                annotation_text="Low Limit",annotation_font=dict(color="#4a9ab5",size=10))
            fig.update_layout(title="Temperature & Humidity",title_font=dict(color="#5a8a68",size=12),**CBASE)
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig2 = make_subplots(specs=[[{"secondary_y":True}]])
            fig2.add_trace(go.Scatter(x=df["step"],y=df["soil_moisture"],name="Soil Moisture (%)",
                line=dict(color="#2d7a4f",width=2.5),fill="tozeroy",fillcolor="rgba(45,122,79,.08)"),secondary_y=False)
            fig2.add_trace(go.Scatter(x=df["step"],y=df["light_level"],name="Light Level (lux)",
                line=dict(color="#c8a040",width=2.5)),secondary_y=True)
            fig2.add_hrect(y0=45,y1=75,fillcolor="rgba(45,122,79,.06)",
                annotation_text="Optimal Soil",annotation_font=dict(color="#2d7a4f",size=10),secondary_y=False)
            fig2.update_layout(title="Soil Moisture & Light Level",title_font=dict(color="#5a8a68",size=12),**CBASE)
            st.plotly_chart(fig2, use_container_width=True)
        with t3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df["step"],y=df["co2_level"],name="CO2 (ppm)",
                line=dict(color="#7c6bb0",width=2.5),fill="tozeroy",fillcolor="rgba(124,107,176,.07)"))
            fig3.add_hrect(y0=600,y1=1200,fillcolor="rgba(45,122,79,.05)",
                annotation_text="Optimal CO2",annotation_font=dict(color="#2d7a4f",size=10))
            fig3.add_hline(y=1200,line_dash="dot",line_color="rgba(200,113,74,.4)",
                annotation_text="High Limit",annotation_font=dict(color="#c8714a",size=10))
            fig3.add_hline(y=600,line_dash="dot",line_color="rgba(200,160,64,.4)",
                annotation_text="Low Limit",annotation_font=dict(color="#c8a040",size=10))
            fig3.update_layout(title="CO2 Concentration",title_font=dict(color="#5a8a68",size=12),**CBASE)
            st.plotly_chart(fig3, use_container_width=True)

    # — Optimization ───────────────────────────────────────────────────────────
    if st.session_state.optimization_result:
        opt = st.session_state.optimization_result
        sec("🧬  Genetic Algorithm — Optimized Irrigation Schedule", "#c8a040")
        o1,o2,o3 = st.columns(3)
        o1.metric("Best Fitness Score",         f"{opt['fitness']:.2f}")
        o2.metric("Irrigation Hours Scheduled", f"{opt['total_irrigations']} / 24 hrs")
        o3.metric("Water Conservation Rate",    f"{(24-opt['total_irrigations'])/24:.1%}")
        sdf = pd.DataFrame({
            "Hour":[f"{h:02d}:00" for h in range(24)],
            "Value":opt["schedule"],
            "Status":["Irrigate" if x else "Skip" for x in opt["schedule"]]})
        fs = px.bar(sdf,x="Hour",y="Value",color="Status",
            color_discrete_map={"Irrigate":"#2d7a4f","Skip":"#eef5ec"},
            title="Optimized 24-Hour Irrigation Schedule",labels={"Value":"Irrigate (1=Yes)"})
        fs.update_layout(height=260,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(240,247,237,1)",font=dict(color="#5a8a68",size=11),
            title_font=dict(color="#5a8a68",size=12),legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=40,b=10),
            xaxis=dict(gridcolor="rgba(45,122,79,0.10)"),
            yaxis=dict(gridcolor="rgba(45,122,79,0.10)"))
        st.plotly_chart(fs, use_container_width=True)
        ff = go.Figure()
        ff.add_trace(go.Scatter(y=opt["best_history"],name="Best Fitness",
            line=dict(color="#2d7a4f",width=2.5),fill="tozeroy",fillcolor="rgba(45,122,79,.08)"))
        ff.add_trace(go.Scatter(y=opt["avg_history"],name="Average Fitness",
            line=dict(color="#c8a040",width=2,dash="dash")))
        ff.update_layout(height=260,title="Fitness Convergence",
            xaxis_title="Generation",yaxis_title="Fitness",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(240,247,237,1)",
            font=dict(color="#5a8a68",size=11),title_font=dict(color="#5a8a68",size=12),
            legend=dict(bgcolor="rgba(0,0,0,0)"),margin=dict(l=10,r=10,t=40,b=10),
            xaxis=dict(gridcolor="rgba(45,122,79,0.10)"),
            yaxis=dict(gridcolor="rgba(45,122,79,0.10)"))
        st.plotly_chart(ff, use_container_width=True)

    # — Action Log ─────────────────────────────────────────────────────────────
    sec("📋  Agent Action Log", "#c8714a")
    if agent.action_log:
        ddf = pd.DataFrame([{
            "Step":int(r["timestamp"]),
            "Action":ACTION_LABELS.get(r["action"],r["action"].replace("_"," ").title()),
            "Reason":r["reason"],"Logic Rule":r["rule"]
        } for r in agent.action_log[-25:]])
        st.dataframe(ddf, use_container_width=True, hide_index=True, column_config={
            "Step":      st.column_config.NumberColumn("Step", width="small"),
            "Action":    st.column_config.TextColumn("Action", width="medium"),
            "Reason":    st.column_config.TextColumn("Reason", width="large"),
            "Logic Rule":st.column_config.TextColumn("Logic Rule", width="large")})
    else:
        st.markdown(card(
            '<div style="text-align:center;color:#c8dfc4;font-size:.82rem;">'
            'No actions logged yet — run a few steps to see agent decisions.</div>',
            padding="22px"), unsafe_allow_html=True)

    # — Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:44px;">
        <div style="height:1px;background:linear-gradient(90deg,
            transparent,#c8e0c4,#e8d8c4,#c8e0c4,transparent);
            margin-bottom:22px;"></div>
        <div style="background:#ffffff;border:1px solid #ddeedd;border-radius:14px;
                    padding:16px 40px;text-align:center;
                    box-shadow:0 1px 8px rgba(45,122,79,0.06);">
            <div style="display:flex;justify-content:center;gap:18px;flex-wrap:wrap;">
                <span style="color:#2d7a4f;font-size:.72rem;letter-spacing:.5px;">🌿 Smart Green House Artificial Intelligence</span>
                <span style="color:#d4e8cc;font-size:.72rem;">·</span>
                <span style="color:#6a9a78;font-size:.72rem;">Principles of Artificial Intelligence</span>
                <span style="color:#d4e8cc;font-size:.72rem;">·</span>
                <span style="color:#4a9ab5;font-size:.72rem;">Logic</span>
                <span style="color:#d4e8cc;font-size:.72rem;">·</span>
                <span style="color:#7c6bb0;font-size:.72rem;">Linear Algebra</span>
                <span style="color:#d4e8cc;font-size:.72rem;">·</span>
                <span style="color:#c8a040;font-size:.72rem;">Probability</span>
                <span style="color:#d4e8cc;font-size:.72rem;">·</span>
                <span style="color:#2d7a4f;font-size:.72rem;">Genetic Algorithm</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
