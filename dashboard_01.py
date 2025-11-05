# build_dashboard_presets_custom_constra_metric_score_251007.py
# One-page, self-contained dashboard with:
# - Two 3D Plotly figures (A: Err–Cov–Energy color=Latency; B: Err–Cov–Latency color=Energy)
# - Presets + Custom weight sliders (10-term score)
# - Top-K selector (30/50) with live overlay of Top-K points
# - Constraints editor (Cov≥, Lat≤, Energy≤) that recomputes ε-knees client-side (ε = 5/10/15%)
# - WScore everywhere: color mode, sidebar panel, knee table, CSV/JSON exports
# - Safe HTML templating via token replacement (no f-strings)

import os, json, numpy as np, pandas as pd
import plotly.graph_objs as go

# ---------------- Configuration ----------------
OUT_DIR = "html"
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate CSV paths (first one that exists is used)

##CANDIDATES = [
##    "outputs/tradeoffs_all_points_EXPLICIT.csv",
##    "outputs/tradeoffs_all_points_EXT.csv",
##]
CANDIDATES = [
    "csvs/tradeoffs_all_points_EXPLICIT.csv",    
]

# Default constraints and epsilon set (client can edit)
COV_MIN = 0.65
LAT_MAX = 4.0         # seconds
ENE_MAX = 300.0       # mWh/day
EPS_LIST = (0.05, 0.10, 0.15)

# 10-term presets (weights will be L1-normalized)
PRESETS = {
    "Balanced":     {"wE":1.0,"wL":1.0,"wEr":1.0,"wC":1.0,"wBD":0.5,"wEDP":0.5,"wEEP":0.5,"wNE":0.5,"wSU":0.5,"wCB":0.5},
    "Product":      {"wE":0.8,"wL":1.4,"wEr":1.0,"wC":1.2,"wBD":0.6,"wEDP":0.4,"wEEP":0.6,"wNE":0.4,"wSU":0.6,"wCB":0.3},
    "Clinical":     {"wE":0.7,"wL":0.9,"wEr":1.7,"wC":1.3,"wBD":0.8,"wEDP":0.6,"wEEP":0.8,"wNE":1.0,"wSU":1.0,"wCB":0.1},
    "BatteryFirst": {"wE":1.4,"wL":0.8,"wEr":0.9,"wC":1.0,"wBD":1.2,"wEDP":0.9,"wEEP":0.9,"wNE":0.3,"wSU":0.5,"wCB":0.2},
    "Research":     {"wE":0.5,"wL":0.6,"wEr":1.6,"wC":1.2,"wBD":0.3,"wEDP":0.4,"wEEP":0.7,"wNE":1.0,"wSU":0.8,"wCB":0.3},
    "CloudFirst":   {"wE":0.6,"wL":0.8,"wEr":1.2,"wC":1.0,"wBD":0.3,"wEDP":0.5,"wEEP":0.7,"wNE":0.8,"wSU":0.5,"wCB":1.2},
}

# 10 metrics: (key, column name, pretty label, normalization mode)
# if mode == "cost": n(x) used; if "benefit_to_cost": use 1 - n(x)
TERMS = [
    ("wE",   "Energy_mWh_day",      "Energy (mWh/day)",   "cost"),
    ("wL",   "Latency_p95_s",       "Latency p95 (s)",    "cost"),
    ("wEr",  "Error",               "Error",              "cost"),
    ("wC",   "Coverage",            "Coverage",           "benefit_to_cost"),  # cost = 1 - n(Coverage)
    ("wBD",  "BatteryDays",         "BatteryDays",        "benefit_to_cost"),  # cost = 1 - n(BatteryDays)
    ("wEDP", "EDP",                 "EDP",                "cost"),
    ("wEEP", "EEprod",              "EEprod",             "cost"),
    ("wNE",  "NormError_%",         "NormError (%)",      "cost"),
    ("wSU",  "StressUplift_%",      "StressUplift (%)",   "cost"),
    ("wCB",  "CloudBoost_pct",      "CloudBoost (%)",     "cost"),
]

# ---------------- Load Data ----------------
for p in CANDIDATES:
    if os.path.exists(p):
        IN_CSV = p
        break
else:
    raise FileNotFoundError("Could not find CSV. Expected one of:\n  " + "\n  ".join(CANDIDATES))

df = pd.read_csv(IN_CSV)

def pick(*names, req=True):
    for n in names:
        if n in df.columns:
            return n
    if req: raise KeyError(f"Missing any of columns: {names}")
    return None

cERR   = pick("Error","err","error")
cCOV   = pick("Coverage","cov","coverage")
cENE   = pick("Energy_mWh_day","Energy","energy_mWh_day")
cLAT   = pick("Latency_p95_s","latency_p95_s","lat_p95")
cMODEL = pick("Model","model", req=False)
cNET   = pick("network","Network","mode", req=False)
cDUTY  = pick("duty","Duty","d", req=False)
cLAYER = pick("DominantLayer","IoT_Layers","IoT_Layer","layer", req=False)
cNAME  = pick("ConfigName","name","id", req=False)

# fill optionals for robust hovers
if cMODEL is None: df["Model"] = "base"; cMODEL = "Model"
if cNET   is None: df["Network"] = "edge"; cNET = "Network"
if cDUTY  is None: df["Duty"] = 1.0; cDUTY = "Duty"
if cLAYER is None: df["IoT_Layers"] = "L1_Device"; cLAYER = "IoT_Layers"
if cNAME  is None: df["ConfigName"] = [f"cfg_{i}" for i in range(len(df))]; cNAME = "ConfigName"

# -------------- Normalization for 10-term score --------------
def n01(series):
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

norm_cols = {}
for key, metric, _, how in TERMS:
    if metric not in df.columns:
        df[metric] = 0.0
    if how == "cost":
        norm_cols[metric] = n01(df[metric])
    else:
        norm_cols[metric] = 1.0 - n01(df[metric])

# convenience arrays
arr = {
    "Error":    df[cERR].to_numpy(float),
    "Coverage": df[cCOV].to_numpy(float),
    "Energy":   df[cENE].to_numpy(float),
    "Latency":  df[cLAT].to_numpy(float),
    "Layer":    df[cLAYER].astype(str).fillna("—").to_numpy(),
    "Name":     df[cNAME].astype(str).to_numpy(),
    "Model":    df[cMODEL].astype(str).to_numpy(),
    "Network":  df[cNET].astype(str).to_numpy(),
    "Duty":     pd.to_numeric(df[cDUTY], errors="coerce").fillna(0).to_numpy(float),
}
N = len(df)

# ----------- Base Plotly figures (static layer traces; overlays updated in JS) -----------
def n01_np(x):
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x)
    return (x - mn)/(mx - mn)

def traces_by_layer(x, y, z, color_vals, colorbar_title, colorscale, hovertext):
    traces = []
    layers = arr["Layer"]
    uniq = list(dict.fromkeys(layers))
    for i, L in enumerate(uniq):
        m = (layers == L)
        traces.append(go.Scatter3d(
            x=x[m], y=y[m], z=z[m], mode="markers", name=str(L),
            marker=dict(
                size=6 + 10*n01_np(arr["Energy"][m]),
                color=color_vals[m], colorscale=colorscale, showscale=(i==0),
                colorbar=dict(title=colorbar_title) if i==0 else None,
                opacity=0.95, line=dict(width=1, color="rgba(0,0,0,0.6)")
            ),
            hoverinfo="text", hovertext=hovertext[m]
        ))
    return traces

def build_hover(score=None, rank=None):
    out = []
    for i in range(N):
        lines = [
            f"<b>{arr['Name'][i]}</b>",
            f"Model: {arr['Model'][i]}",
            f"Network: {arr['Network'][i]}",
            f"Duty: {arr['Duty'][i]:.2f}",
            f"Layer: {arr['Layer'][i]}",
            f"Error: {arr['Error'][i]:.3g}",
            f"Coverage: {arr['Coverage'][i]:.3g}",
            f"Latency p95 (s): {arr['Latency'][i]:.3g}",
            f"Energy (mWh/day): {arr['Energy'][i]:.3g}",
        ]
        if score is not None and rank is not None:
            lines += [f"WScore: {score[i]:.3f}", f"Rank(Custom): {int(rank[i])}"]
        out.append("<br>".join(lines))
    return np.array(out, dtype=object)

base_hover = build_hover()

# Figure A (Err–Cov–Energy, color=Latency)
xA, yA, zA, cA = arr["Error"], arr["Coverage"], arr["Energy"], arr["Latency"]
tracesA = traces_by_layer(xA, yA, zA, cA, "Latency p95 (s)", "Viridis", base_hover)
# placeholders for ε-knees and TopK overlays (populated in JS)
tracesA.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="ε-knees",
                            marker=dict(size=10, color="white", opacity=1.0, line=dict(width=2.5, color="black")),
                            hoverinfo="text", hovertext=[]))
tracesA.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="TopK Custom",
                            marker=dict(size=12, color="white", opacity=1.0, line=dict(width=3, color="black")),
                            hoverinfo="text", hovertext=[]))
figA = go.Figure(data=tracesA)
figA.update_layout(
    title="A. Error–Coverage–Energy (color = Latency)",
    scene=dict(xaxis_title="Error", yaxis_title="Coverage", zaxis_title="Energy (mWh/day)"),
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=0, r=0, t=60, b=0)
)

# Figure B (Err–Cov–Latency, color=Energy)
xB, yB, zB, cB = arr["Error"], arr["Coverage"], arr["Latency"], arr["Energy"]
tracesB = traces_by_layer(xB, yB, zB, cB, "Energy (mWh/day)", "Plasma", base_hover)
tracesB.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="ε-knees",
                            marker=dict(size=10, color="white", opacity=1.0, line=dict(width=2.5, color="black")),
                            hoverinfo="text", hovertext=[]))
tracesB.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="TopK Custom",
                            marker=dict(size=12, color="white", opacity=1.0, line=dict(width=3, color="black")),
                            hoverinfo="text", hovertext=[]))
figB = go.Figure(data=tracesB)
figB.update_layout(
    title="B. Error–Coverage–Latency (color = Energy)",
    scene=dict(xaxis_title="Error", yaxis_title="Coverage", zaxis_title="Latency p95 (s)"),
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=0, r=0, t=60, b=0)
)

# HTML fragments
figA_div = figA.to_html(full_html=False, include_plotlyjs='cdn')
figB_div = figB.to_html(full_html=False, include_plotlyjs=False)

# Knee table (initial shell; rows filled in JS)
knee_html = (
    '<table class="table table-sm table-striped knee"><thead><tr>'
    '<th>ConfigName</th><th>Model</th><th>Network</th><th>Duty</th><th>IoT_Layers</th>'
    '<th>Error</th><th>Coverage</th><th>Latency_p95_s</th><th>Energy_mWh_day</th>'
    '</tr></thead><tbody></tbody></table>'
)

# Normalize presets L1
def normalize_weights(w):
    v = [float(w[k]) for (k,_,_,_) in TERMS]
    s = sum(abs(x) for x in v)
    if s <= 0:
        return {k: 1.0/len(TERMS) for (k,_,_,_) in TERMS}
    return {k: float(w[k])/s for (k,_,_,_) in TERMS}

PRESETS_N = {p: normalize_weights(w) for p, w in PRESETS.items()}

# Pack normalized columns and raw data for JS
norm_pack = {metric: norm_cols[metric].astype(float).tolist() for _, metric, _, _ in TERMS}
raw_pack = {
    "Error":    arr["Error"].astype(float).tolist(),
    "Coverage": arr["Coverage"].astype(float).tolist(),
    "Energy":   arr["Energy"].astype(float).tolist(),
    "Latency":  arr["Latency"].astype(float).tolist(),
    "Layer":    arr["Layer"].tolist(),
    "Name":     arr["Name"].tolist(),
    "Model":    arr["Model"].tolist(),
    "Network":  arr["Network"].tolist(),
    "Duty":     arr["Duty"].astype(float).tolist(),
}

# ---------- HTML (token-replacement; no f-strings) ----------
eps_list_pct = ",".join([str(int(x*100)) for x in EPS_LIST])
preset_options = "".join(f'<option value="{p}">{p}</option>' for p in PRESETS_N.keys())
TERMS_JSON = json.dumps([k for (k,_,_,_) in TERMS])
TERM2METRIC_JSON = json.dumps({k: m for (k, m, _, _) in TERMS})
PRESETS_N_JSON = json.dumps(PRESETS_N)
NORM_PACK_JSON = json.dumps(norm_pack)
RAW_PACK_JSON  = json.dumps(raw_pack)

html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pareto Dashboard — Presets + Custom + Constraints + WScore</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --panel-w: 360px; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
.header { padding:16px 24px; background:#0b3d91; color:#fff; position:sticky; top:0; z-index:5; }
.summary { color:#e8eefc; font-size:14px; }
.layout { display:flex; flex-direction:row; align-items:flex-start; }
.main { flex:1; padding:16px 24px; max-width: calc(100% - var(--panel-w)); }
.panel {
  width: var(--panel-w); border-left:1px solid #e2e2e2; background:#fbfbfc;
  padding:12px 12px; position:sticky; top:64px; height: calc(100vh - 64px); overflow:auto;
}
.card { background:#fff; border:1px solid #e2e2e2; border-radius:8px; margin-bottom:16px; }
.card .card-header { padding:12px 16px; font-weight:600; border-bottom:1px solid #eee; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.card .card-body { padding:16px; }
.smallcode { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12px; background:#f6f8fa; padding:2px 4px; border-radius:4px; color:#111; }
.btn { display:inline-block; padding:6px 10px; border:1px solid #ddd; background:#f6f8fa; border-radius:6px; cursor:pointer; font-size:13px; }
select, input[type="number"] { padding:6px 8px; border:1px solid #ddd; border-radius:6px; background:#fff; font-size:13px; }
.table-container { overflow:auto; max-height: 420px; border:1px solid #eee; border-radius:8px; }
.table { border-collapse: collapse; width:100%; font-size:13px; }
.table th, .table td { padding:6px 10px; border-bottom:1px solid #eee; white-space:nowrap; }
.section-title { font-size:12px; font-weight:700; letter-spacing:.04em; color:#555; margin-top:8px; margin-bottom:6px; text-transform:uppercase; }
.slider-row { margin:8px 0; }
.slider-row label { display:block; font-size:12px; color:#444; margin-bottom:4px; }
.slider-row input[type="range"] { width:100%; }
.form-grid { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
.help { font-size:12px; color:#666; }
.hr { height:1px; background:#eee; margin:10px 0; }
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <div class="header">
    <h1 style="margin:0;">Pareto Dashboard — Presets + Custom + Constraints + WScore</h1>
    <div class="summary">
      ε default: __EPS_LIST_PCT__ &nbsp; | &nbsp;
      Source: <span class="smallcode">__IN_CSV__</span>
    </div>
    <div style="margin-top:8px;">
      <button class="btn" id="openLegend">Metrics Legend</button>
    </div>
  </div>

  <div class="layout">
    <div class="main">

      <!-- Color mode selector -->
      <div class="card" style="margin-bottom:12px;">
        <div class="card-body">
          <div class="controls" style="gap:16px;align-items:center;">
            <div style="font-weight:600;">Color mode:</div>
            <label><input type="radio" name="colorMode" value="Latency" checked> Latency</label>
            <label><input type="radio" name="colorMode" value="Energy"> Energy</label>
            <label><input type="radio" name="colorMode" value="WScore"> WScore</label>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" style="gap:8px;">
          <div>A. Error–Coverage–Energy (color = Latency / Energy / WScore)</div>
          <div style="margin-left:auto;display:flex;gap:8px;">
            <button class="btn" id="dlPNGA">Download PNG (Fig A)</button>
          </div>
        </div>
        <div class="card-body">
          <div id="figA">__FIG_A__</div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" style="gap:8px;">
          <div>B. Error–Coverage–Latency (color = Energy / Latency / WScore)</div>
          <div style="margin-left:auto;display:flex;gap:8px;">
            <button class="btn" id="dlPNGB">Download PNG (Fig B)</button>
          </div>
        </div>
        <div class="card-body">
          <div id="figB">__FIG_B__</div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" style="gap:8px;">
          <div>ε-Knee Candidates</div>
          <div class="help">Filter/search; reflects current constraints.</div>
          <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
            <input id="search" type="text" placeholder="Search knees table...">
            <label style="font-size:12px;">Top-K:
              <select id="topkSel">
                <option value="30">30</option>
                <option value="50" selected>50</option>
              </select>
            </label>
            <label style="font-size:12px;"><input type="checkbox" id="topkOnly"> Show Top-K only</label>
            <button class="btn" id="dlKnees">Download knees CSV</button>
            <button class="btn" id="dlTopk">Download Top-K CSV</button>
            <button class="btn" id="exportJSON">Export Snapshot (JSON)</button>
          </div>
        </div>
        <div class="card-body">
          <div class="table-container" id="tableWrap">
            __KNEE_TABLE__
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="section-title">Preset</div>
      <select id="presetSel" style="width:100%; margin-bottom:8px;">
        __PRESET_OPTIONS__
        <option value="__CUSTOM__">Custom (sliders)</option>
      </select>

      <div class="hr"></div>
      <div class="section-title">Custom Weights (L1-normalized)</div>
      <div id="sliders"></div>
      <div class="help">Drag sliders to enter <b>Custom</b> mode; Top-K and overlays update live.</div>

      <div class="hr"></div>
      <div class="section-title">Current Weighted Score</div>
      <div id="scoreBox" class="card">
        <div class="card-body" style="text-align:center;">
          <div id="scoreValue" style="font-size:32px; font-weight:800; color:#0a66c2;">–</div>
          <div class="help">Mean composite cost across all configurations (lower = better)</div>
        </div>
      </div>

      <div class="hr"></div>
      <div class="section-title">Constraints & ε</div>
      <div class="form-grid">
        <label>Coverage ≥ <input id="covMin" type="number" step="0.01" value="__COV_MIN__"></label>
        <label>Latency ≤ (s) <input id="latMax" type="number" step="0.1" value="__LAT_MAX__"></label>
        <label>Energy ≤ (mWh/day) <input id="eneMax" type="number" step="10" value="__ENE_MAX__"></label>
        <label>ε (%) <input id="epsList" type="text" value="__EPS_LIST_PCT__"></label>
      </div>
      <div style="margin-top:8px;display:flex;gap:8px;">
        <button class="btn" id="applyC">Apply</button>
        <button class="btn" id="resetC">Reset</button>
      </div>
      <div class="help">Format for ε: e.g., <span class="smallcode">5,10,15</span></div>

      <div class="hr"></div>
      <div class="section-title">Metrics Legend</div>
      <details id="legend" class="card">
        <summary class="card-header">Show / Hide</summary>
        <div class="card-body">
          <div class="help" style="margin-bottom:8px;">
            Directions indicate the optimization direction used in scoring; items marked (1−n) are converted to a cost by 1 − normalized(value).
          </div>
          <table class="table">
            <thead>
              <tr><th>Key</th><th>Metric</th><th>Meaning</th><th>Direction</th></tr>
            </thead>
            <tbody>
              <tr><td><code>wEr</code></td><td>Error</td><td>Primary estimation/measurement error</td><td>↓ minimize</td></tr>
              <tr><td><code>wC</code></td><td>Coverage</td><td>Wear-time / usable proportion</td><td>↑ maximize (scored 1−n)</td></tr>
              <tr><td><code>wL</code></td><td>Latency p95 (s)</td><td>95th-percentile latency</td><td>↓ minimize</td></tr>
              <tr><td><code>wE</code></td><td>Energy (mWh/day)</td><td>Daily energy budget</td><td>↓ minimize</td></tr>
              <tr><td><code>wBD</code></td><td>BatteryDays</td><td>Expected endurance</td><td>↑ maximize (scored 1−n)</td></tr>
              <tr><td><code>wEDP</code></td><td>EDP</td><td>Energy-Delay Product proxy</td><td>↓ minimize</td></tr>
              <tr><td><code>wEEP</code></td><td>EEprod</td><td>Energy×Error product proxy</td><td>↓ minimize</td></tr>
              <tr><td><code>wNE</code></td><td>NormError (%)</td><td>Normalized error</td><td>↓ minimize</td></tr>
              <tr><td><code>wSU</code></td><td>StressUplift (%)</td><td>Error increase under stressors</td><td>↓ minimize</td></tr>
              <tr><td><code>wCB</code></td><td>CloudBoost (%)</td><td>Cloud-assisted gain</td><td>↓ minimize (as modeled)</td></tr>
            </tbody>
          </table>
          <div class="help" style="margin-top:8px;">
            Scoring uses L1-normalized weights and per-metric normalization to [0,1]. “Benefit_to_cost” metrics are inverted so lower is always better for the linear cost.
          </div>
        </div>
      </details>
    </div>
  </div>

<script>
/* ===========================
   Boot data (injected tokens)
   =========================== */
const TERMS = __TERMS_JSON__;
const TERM2METRIC = __TERM2METRIC_JSON__;
const PRESETS = __PRESETS_N_JSON__;  // normalized weights
const norm = __NORM_PACK_JSON__;
const raw  = __RAW_PACK_JSON__;

let TOPK = 50; // default; user selectable (30/50)
let weights = Object.assign({}, PRESETS["Balanced"]); // current weight vector (normalized)
let customMode = false;

let constraints = {
  covMin: __COV_MIN__,
  latMax: __LAT_MAX__,
  eneMax: __ENE_MAX__,
  eps: [__EPS_LIST_PCT__]
};

/* ================ Utilities ================ */
const N = raw.Name.length;
function $(sel){ return document.querySelector(sel); }
function $all(sel){ return Array.from(document.querySelectorAll(sel)); }

function l1Normalize(obj) {
  let s = 0.0; for (const k of TERMS) s += Math.abs(+obj[k]||0);
  if (s <= 0) { const eq=1.0/TERMS.length; let out={}; for(const k of TERMS) out[k]=eq; return out; }
  let out={}; for (const k of TERMS) out[k] = (+obj[k]||0)/s; return out;
}

/* WScore = Σ w_k * n(metric_k) */
function computeWScore(w) {
  const s = new Float64Array(N);
  for (let i=0;i<N;i++) s[i]=0.0;
  for (const k of TERMS) {
    const metric = TERM2METRIC[k];
    const v = norm[metric];
    const wk = +w[k] || 0;
    for (let i=0;i<N;i++) s[i] += wk * v[i];
  }
  return s;
}

function argsortAsc(a) { return Array.from(a.keys()).sort((i,j)=>a[i]-a[j]); }
function topKIndices(scores, K) { return argsortAsc(scores).slice(0, K); }
function makeHover(i, s=null, r=null) {
  const lines = [
    `<b>${raw.Name[i]}</b>`,
    `Model: ${raw.Model[i]}`,
    `Network: ${raw.Network[i]}`,
    `Duty: ${(+raw.Duty[i]).toFixed(2)}`,
    `Layer: ${raw.Layer[i]}`,
    `Error: ${(+raw.Error[i]).toPrecision(3)}`,
    `Coverage: ${(+raw.Coverage[i]).toPrecision(3)}`,
    `Latency p95 (s): ${(+raw.Latency[i]).toPrecision(3)}`,
    `Energy (mWh/day): ${(+raw.Energy[i]).toPrecision(3)}`
  ];
  if (s && r) { lines.push(`WScore: ${s[i].toFixed(3)}`); lines.push(`Rank(Custom): ${r[i]}`); }
  return lines.join('<br>');
}

/* ===========================
   ε-knees (client-side)
   =========================== */
function computeKnees(covMin, latMax, eneMax, epsPercentList) {
  const feas = [];
  let errMin = +Infinity, eneMin = +Infinity;
  for (let i=0;i<N;i++) {
    if (raw.Coverage[i] >= covMin && raw.Latency[i] <= latMax && raw.Energy[i] <= eneMax) {
      feas.push(i);
      if (raw.Error[i]  < errMin) errMin = raw.Error[i];
      if (raw.Energy[i] < eneMin) eneMin = raw.Energy[i];
    }
  }
  if (feas.length === 0) return [];
  const idxs = new Set();
  for (const e of epsPercentList) {
    const eps = (+e)/100.0;
    const thrErr = (1.0+eps)*errMin;
    const thrEne = (1.0+eps)*eneMin;

    // minimize Energy subject to Error ≤ thrErr
    let bestE = +Infinity, bestA = -1;
    for (const i of feas) {
      if (raw.Error[i] <= thrErr && raw.Energy[i] < bestE) { bestE = raw.Energy[i]; bestA = i; }
    }
    if (bestA >= 0) idxs.add(bestA);

    // minimize Error subject to Energy ≤ thrEne
    let bestEr = +Infinity, bestB = -1;
    for (const i of feas) {
      if (raw.Energy[i] <= thrEne && raw.Error[i] < bestEr) { bestEr = raw.Error[i]; bestB = i; }
    }
    if (bestB >= 0) idxs.add(bestB);
  }
  return Array.from(idxs).sort((a,b)=>a-b);
}

/* ===========================
   Table helpers (WScore column)
   =========================== */
function ensureWScoreHeader() {
  const hdr = document.querySelector('.table.knee thead tr');
  if (!hdr) return;
  if (!hdr.querySelector('.wscore')) {
    const th = document.createElement('th');
    th.className = 'wscore';
    th.textContent = 'WScore';
    hdr.appendChild(th);
  }
}
function renderKneesTable(idxList, scores=null) {
  ensureWScoreHeader();
  const tbody = document.querySelector('.table.knee tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  for (const i of idxList) {
    const tr = document.createElement('tr');
    const cells = [
      raw.Name[i], raw.Model[i], raw.Network[i], (+raw.Duty[i]).toFixed(2), raw.Layer[i],
      (+raw.Error[i]).toPrecision(3), (+raw.Coverage[i]).toPrecision(3),
      (+raw.Latency[i]).toPrecision(3), (+raw.Energy[i]).toPrecision(3),
      (scores ? scores[i].toFixed(4) : '')
    ];
    for (const c of cells) {
      const td = document.createElement('td'); td.textContent = c; tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}
function applySearchFilter() {
  const input = $('#search'); if (!input) return;
  const filter = (input.value || '').toLowerCase();
  const trs = $all('.table.knee tbody tr');
  trs.forEach(tr => {
    const text = tr.innerText.toLowerCase();
    tr.style.display = text.indexOf(filter) > -1 ? '' : 'none';
  });
}
function filterTableToIndices(idxSet) {
  const table = document.querySelector('.table.knee'); if (!table) return;
  const trs = table.querySelectorAll('tbody tr');
  const heads = Array.from(table.querySelectorAll('thead th')).map(th=>th.innerText.trim());
  const nameCol = heads.indexOf('ConfigName');
  trs.forEach(tr => {
    const name = tr.children[nameCol]?.innerText.trim() || '';
    const i = raw.Name.indexOf(name);
    tr.style.display = (i>=0 && idxSet.has(i)) ? '' : 'none';
  });
}
function resetTableFilter() { $all('.table.knee tbody tr').forEach(tr => tr.style.display=''); }

/* ===========================
   Plot recoloring & overlays
   =========================== */
function getPlot(ref){ return document.querySelector(ref+' .js-plotly-plot'); }
function hueForScore(x) {
  const v = Math.max(0, Math.min(1, x)); // clamp if needed
  const hue = 120 - v*120;               // 120=green .. 0=red
  return `hsl(${hue},70%,40%)`;
}
function updateScoreBox(scores) {
  const mean = scores.reduce((a,b)=>a+b,0)/scores.length;
  const box = document.getElementById('scoreValue');
  if (box) { box.textContent = mean.toFixed(3); box.style.color = hueForScore(mean); }
}
function recolorPlots(mode, scores=null) {
  const gdA = getPlot('#figA');
  const gdB = getPlot('#figB');
  let cvec, title;
  if (mode === 'WScore') { cvec = scores ? Array.from(scores) : Array(N).fill(0); title = "Weighted Score (lower=better)"; }
  else if (mode === 'Energy') { cvec = Array.from(raw.Energy); title = "Energy (mWh/day)"; }
  else { cvec = Array.from(raw.Latency); title = "Latency p95 (s)"; }

  if (gdA) {
    const layerCount = gdA.data.length - 2; // last two traces are overlays
    for (let t=0; t<layerCount; t++) Plotly.restyle(gdA, {'marker.color':[cvec]}, [t]);
    gdA.layout.scene.colorbar.title.text = title; Plotly.update(gdA);
  }
  if (gdB) {
    const layerCount = gdB.data.length - 2;
    for (let t=0; t<layerCount; t++) Plotly.restyle(gdB, {'marker.color':[cvec]}, [t]);
    gdB.layout.scene.colorbar.title.text = title; Plotly.update(gdB);
  }
}

/* ===========================
   Central refresh
   =========================== */
function updateOverlaysAndTable() {
  weights = l1Normalize(weights);
  const s = computeWScore(weights);              // WScore per config
  updateScoreBox(s);

  const order = argsortAsc(s);
  const ranks = new Array(N); for (let r=0;r<N;r++) ranks[order[r]] = r+1;
  const top = topKIndices(s, TOPK);
  const knees = computeKnees(constraints.covMin, constraints.latMax, constraints.eneMax, constraints.eps);

  // Update overlays in Fig A
  const gdA = getPlot('#figA');
  if (gdA && gdA.data && gdA.data.length >= 2) {
    const idxKneeA = gdA.data.length - 2;
    const idxTopA  = gdA.data.length - 1;
    Plotly.restyle(gdA, {
      x: [knees.map(i=>raw.Error[i])],
      y: [knees.map(i=>raw.Coverage[i])],
      z: [knees.map(i=>raw.Energy[i])],
      hovertext: [knees.map(i=>makeHover(i))]
    }, [idxKneeA]);
    Plotly.restyle(gdA, {
      x: [top.map(i=>raw.Error[i])],
      y: [top.map(i=>raw.Coverage[i])],
      z: [top.map(i=>raw.Energy[i])],
      hovertext: [top.map(i=>makeHover(i, s, ranks))]
    }, [idxTopA]);
  }

  // Update overlays in Fig B
  const gdB = getPlot('#figB');
  if (gdB && gdB.data && gdB.data.length >= 2) {
    const idxKneeB = gdB.data.length - 2;
    const idxTopB  = gdB.data.length - 1;
    Plotly.restyle(gdB, {
      x: [knees.map(i=>raw.Error[i])],
      y: [knees.map(i=>raw.Coverage[i])],
      z: [knees.map(i=>raw.Latency[i])],
      hovertext: [knees.map(i=>makeHover(i))]
    }, [idxKneeB]);
    Plotly.restyle(gdB, {
      x: [top.map(i=>raw.Error[i])],
      y: [top.map(i=>raw.Coverage[i])],
      z: [top.map(i=>raw.Latency[i])],
      hovertext: [top.map(i=>makeHover(i, s, ranks))]
    }, [idxTopB]);
  }

  // Rebuild knee table & filters
  renderKneesTable(knees, s);
  const topkOnly = $('#topkOnly');
  if (topkOnly && topkOnly.checked) { filterTableToIndices(new Set(top)); }
  applySearchFilter();

  // Honor current color mode
  const mode = document.querySelector('input[name="colorMode"]:checked')?.value || 'Latency';
  recolorPlots(mode, s);
}

/* ===========================
   UI wiring
   =========================== */
function buildSliders() {
  const host = $('#sliders'); if (!host) return;
  host.innerHTML = '';
  const labels = {
    "wE":"Energy", "wL":"Latency", "wEr":"Error", "wC":"Coverage (1-n)",
    "wBD":"BatteryDays (1-n)", "wEDP":"EDP", "wEEP":"EEprod",
    "wNE":"NormError%", "wSU":"StressUplift%", "wCB":"CloudBoost%"
  };
  for (const k of TERMS) {
    const row = document.createElement('div'); row.className='slider-row';
    const lab = document.createElement('label'); lab.textContent = labels[k] + ' — weight';
    const rng = document.createElement('input'); rng.type='range'; rng.min='0'; rng.max='2.0'; rng.step='0.05'; rng.value=(weights[k]||0);
    const num = document.createElement('input'); num.type='number'; num.min='0'; num.max='2.0'; num.step='0.05'; num.value=(weights[k]||0); num.style.width='64px'; num.style.marginLeft='6px';
    function onchg(v) {
      customMode = true; const ps = $('#presetSel'); if (ps) ps.value='__CUSTOM__';
      weights[k] = parseFloat(v)||0; rng.value=weights[k]; num.value=weights[k];
      updateOverlaysAndTable();
    }
    rng.addEventListener('input', e=>onchg(e.target.value));
    num.addEventListener('input', e=>onchg(e.target.value));
    row.appendChild(lab); row.appendChild(rng); row.appendChild(num); host.appendChild(row);
  }
}
function setPreset(name) {
  if (name==='__CUSTOM__') { customMode = true; }
  else { customMode=false; weights = Object.assign({}, PRESETS[name]); }
  buildSliders();
  updateOverlaysAndTable();
}
function attachConstraintHandlers() {
  const covMin = $('#covMin'), latMax=$('#latMax'), eneMax=$('#eneMax'), epsList=$('#epsList');
  const applyBtn = $('#applyC'), resetBtn = $('#resetC');
  if (applyBtn) applyBtn.addEventListener('click', ()=>{
    constraints.covMin = parseFloat(covMin.value)||0;
    constraints.latMax = parseFloat(latMax.value)||0;
    constraints.eneMax = parseFloat(eneMax.value)||0;
    constraints.eps = (epsList.value||'').split(',').map(s=>parseFloat(s.trim())).filter(x=>!isNaN(x));
    updateOverlaysAndTable();
  });
  if (resetBtn) resetBtn.addEventListener('click', ()=>{
    covMin.value = __COV_MIN__; latMax.value = __LAT_MAX__; eneMax.value = __ENE_MAX__;
    epsList.value = "__EPS_LIST_PCT__";
    constraints = { covMin: __COV_MIN__, latMax: __LAT_MAX__, eneMax: __ENE_MAX__, eps: [__EPS_LIST_PCT__] };
    updateOverlaysAndTable();
  });
}
function attachSearchAndTopK() {
  const q = $('#search'); if (q) q.addEventListener('keyup', applySearchFilter);
  const only = $('#topkOnly'); if (only) only.addEventListener('change', ()=>{
    if (only.checked) {
      const s = computeWScore(l1Normalize(weights));
      const top = topKIndices(s, TOPK);
      filterTableToIndices(new Set(top));
      applySearchFilter();
    } else {
      resetTableFilter(); applySearchFilter();
    }
  });
  const topSel = $('#topkSel'); if (topSel) topSel.addEventListener('change', (e)=>{
    TOPK = parseInt(e.target.value)||50;
    updateOverlaysAndTable();
  });
}
function downloadVisibleTable(filename) {
  const table = document.querySelector('.table.knee'); if (!table) return;
  const rows = table.querySelectorAll('tr');
  const data = [];
  rows.forEach(row => {
    if (row.style.display==='none') return;
    const cols = row.querySelectorAll('th,td');
    const rowData = [];
    cols.forEach(cell => {
      let t = (cell.innerText||'').replaceAll('"','""');
      if (t.indexOf(',')>=0 || t.indexOf('"')>=0) t = '"' + t + '"';
      rowData.push(t);
    });
    data.push(rowData.join(','));
  });
  const blob = new Blob([data.join('\n')],{type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function downloadCSV(rows, filename) {
  const escaped = rows.map(r => r.map(v => {
    v = String(v).replaceAll('"','""');
    return (v.indexOf(',')>=0 || v.indexOf('"')>=0) ? ('"'+v+'"') : v;
  }).join(','));
  const blob = new Blob([escaped.join('\n')], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function downloadPlotPNG(selector, filename) {
  const gd = document.querySelector(selector+' .js-plotly-plot');
  if (!gd) { alert('Plot not found for '+selector); return; }
  Plotly.downloadImage(gd, {format:'png', filename: filename.replace('.png',''), height:900, width:1100, scale:2});
}
function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj,null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function attachDownloads() {
  const dlKnees = $('#dlKnees'); if (dlKnees) dlKnees.addEventListener('click', ()=>downloadVisibleTable('eps_knees_filtered.csv'));
  const dlTopk  = $('#dlTopk');  if (dlTopk ) dlTopk .addEventListener('click', ()=>{
    const wN = l1Normalize(weights);
    const s = computeWScore(wN);
    const order = argsortAsc(s);
    const ranks = new Array(N); for (let r=0;r<N;r++) ranks[order[r]]=r+1;
    const top = topKIndices(s, TOPK);
    const rows = [['ConfigName','Model','Network','Duty','IoT_Layers','Error','Coverage','Latency_p95_s','Energy_mWh_day','WScore','Rank_Custom']];
    for (const i of top) {
      rows.push([
        raw.Name[i], raw.Model[i], raw.Network[i], (+raw.Duty[i]).toFixed(2), raw.Layer[i],
        (+raw.Error[i]).toPrecision(3), (+raw.Coverage[i]).toPrecision(3),
        (+raw.Latency[i]).toPrecision(3), (+raw.Energy[i]).toPrecision(3),
        s[i].toFixed(4), ranks[i]
      ]);
    }
    downloadCSV(rows, 'topK_custom.csv');
  });
  const dlA = $('#dlPNGA'); if (dlA) dlA.addEventListener('click', ()=>downloadPlotPNG('#figA','figure_A.png'));
  const dlB = $('#dlPNGB'); if (dlB) dlB.addEventListener('click', ()=>downloadPlotPNG('#figB','figure_B.png'));
  const expJ = $('#exportJSON'); if (expJ) expJ.addEventListener('click', ()=>{
    const snapshot = {
      weights: l1Normalize(weights),
      constraints: constraints,
      eps: constraints.eps,
      topK: TOPK,
      timestamp: new Date().toISOString()
    };
    const wN = l1Normalize(weights);
    const s = computeWScore(wN);
    const top = topKIndices(s, TOPK);
    snapshot.topIndices = top;
    snapshot.knees = computeKnees(constraints.covMin, constraints.latMax, constraints.eneMax, constraints.eps);
    snapshot.wscoreMean = s.reduce((a,b)=>a+b,0)/s.length;
    downloadJSON(snapshot, 'snapshot.json');
  });
}
function attachColorMode() {
  $all('input[name="colorMode"]').forEach(r=>{
    r.addEventListener('change', ()=>{
      const s = computeWScore(l1Normalize(weights));
      const mode = document.querySelector('input[name="colorMode"]:checked')?.value || 'Latency';
      recolorPlots(mode, s);
    });
  });
}
function initPresetSelector(){
  const ps = $('#presetSel'); if (!ps) return;
  ps.addEventListener('change', (e)=>setPreset(e.target.value));
}
function initLegendButton(){
  const btn = document.getElementById('openLegend');
  const det = document.getElementById('legend');
  if (btn && det) btn.addEventListener('click', ()=>{ det.open = true; det.scrollIntoView({behavior:'smooth', block:'start'}); });
}

/* ===========================
   Init
   =========================== */
document.addEventListener('DOMContentLoaded', () => {
  buildSliders();
  attachConstraintHandlers();
  attachSearchAndTopK();
  attachDownloads();
  attachColorMode();
  initPresetSelector();
  updateOverlaysAndTable(); // computes WScore, updates colors/overlays/table
  initLegendButton();
});
</script>
</body>
</html>
"""

# Do the token replacements
html = (html_template
        .replace("__EPS_LIST_PCT__", eps_list_pct)
        .replace("__IN_CSV__", IN_CSV)
        .replace("__FIG_A__", figA_div)
        .replace("__FIG_B__", figB_div)
        .replace("__KNEE_TABLE__", knee_html)
        .replace("__PRESET_OPTIONS__", preset_options)
        .replace("__COV_MIN__", str(COV_MIN))
        .replace("__LAT_MAX__", str(LAT_MAX))
        .replace("__ENE_MAX__", str(ENE_MAX))
        .replace("__TERMS_JSON__", TERMS_JSON)
        .replace("__TERM2METRIC_JSON__", TERM2METRIC_JSON)
        .replace("__PRESETS_N_JSON__", PRESETS_N_JSON)
        .replace("__NORM_PACK_JSON__", NORM_PACK_JSON)
        .replace("__RAW_PACK_JSON__", RAW_PACK_JSON)
        )

out_html = os.path.join(OUT_DIR, "pareto_dashboard_presets_custom_constraints_score.html")
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)

print("Dashboard written to:", out_html)
