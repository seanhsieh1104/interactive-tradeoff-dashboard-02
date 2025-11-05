
# build_dashboard_presets_custom_constra_metric_score_with_tab_251007.py
# Integrated version: adds "Scores Analytics" tab switcher to the existing Pareto dashboard

import os, json, numpy as np, pandas as pd, plotly.graph_objs as go

OUT_DIR = "html"
os.makedirs(OUT_DIR, exist_ok=True)

##CANDIDATES = [
##    "outputs/tradeoffs_all_points_EXPLICIT.csv",
##    "outputs/tradeoffs_all_points_EXT.csv",
##]
CANDIDATES = [
    "csvs/tradeoffs_all_points_EXPLICIT.csv",    
]

for p in CANDIDATES:
    if os.path.exists(p):
        IN_CSV = p
        break
else:
    raise FileNotFoundError("Expected one of: " + ", ".join(CANDIDATES))

df = pd.read_csv(IN_CSV)

# ---------------- Configuration ----------------
TERMS = [
    ("wE",   "Energy_mWh_day",      "Energy (mWh/day)",   "cost"),
    ("wL",   "Latency_p95_s",       "Latency p95 (s)",    "cost"),
    ("wEr",  "Error",               "Error",              "cost"),
    ("wC",   "Coverage",            "Coverage",           "benefit_to_cost"),
    ("wBD",  "BatteryDays",         "BatteryDays",        "benefit_to_cost"),
    ("wEDP", "EDP",                 "EDP",                "cost"),
    ("wEEP", "EEprod",              "EEprod",             "cost"),
    ("wNE",  "NormError_%",         "NormError (%)",      "cost"),
    ("wSU",  "StressUplift_%",      "StressUplift (%)",   "cost"),
    ("wCB",  "CloudBoost_pct",      "CloudBoost (%)",     "cost"),
]

PRESETS = {
    "Balanced": {"wE":1.0,"wL":1.0,"wEr":1.0,"wC":1.0,"wBD":0.5,"wEDP":0.5,"wEEP":0.5,"wNE":0.5,"wSU":0.5,"wCB":0.5},
    "Research": {"wE":0.5,"wL":0.6,"wEr":1.6,"wC":1.2,"wBD":0.3,"wEDP":0.4,"wEEP":0.7,"wNE":1.0,"wSU":0.8,"wCB":0.3},
    "Clinical": {"wE":0.7,"wL":0.9,"wEr":1.7,"wC":1.3,"wBD":0.8,"wEDP":0.6,"wEEP":0.8,"wNE":1.0,"wSU":1.0,"wCB":0.1},
    "Product":  {"wE":0.8,"wL":1.4,"wEr":1.0,"wC":1.2,"wBD":0.6,"wEDP":0.4,"wEEP":0.6,"wNE":0.4,"wSU":0.6,"wCB":0.3},
    "BatteryFirst": {"wE":1.4,"wL":0.8,"wEr":0.9,"wC":1.0,"wBD":1.2,"wEDP":0.9,"wEEP":0.9,"wNE":0.3,"wSU":0.5,"wCB":0.2},
    "CloudFirst": {"wE":0.6,"wL":0.8,"wEr":1.2,"wC":1.0,"wBD":0.3,"wEDP":0.5,"wEEP":0.7,"wNE":0.8,"wSU":0.5,"wCB":1.2},
}

def n01(series):
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

# normalize all metrics
norm_cols = {}
for key, metric, _, how in TERMS:
    if metric not in df.columns:
        df[metric] = 0.0
    if how == "cost":
        norm_cols[metric] = n01(df[metric])
    else:
        norm_cols[metric] = 1.0 - n01(df[metric])

def normalize_weights(w):
    v = np.array(list(w.values()), float)
    s = np.sum(np.abs(v))
    if s <= 0: return {k:1.0/len(v) for k in w}
    return {k: float(w[k])/s for k in w}

PRESETS_N = {p: normalize_weights(w) for p, w in PRESETS.items()}

# compute WScore for each preset
scores = {}
for pname, w in PRESETS_N.items():
    cost = np.zeros(len(df))
    for k,m,_,_ in TERMS:
        cost += w[k]*norm_cols[m].to_numpy(float)
    scores[pname] = cost

# build score summary table
rank_idx = np.argsort(scores["Research"])
top = df.loc[rank_idx[:12]].copy()
top["WScore"] = scores["Research"][rank_idx[:12]]

# --- Plotly figs ---
import plotly.graph_objects as go
fig_bar = go.Figure(go.Bar(x=top["WScore"], y=top["ConfigName"] if "ConfigName" in top else top.index.astype(str), orientation="h"))
fig_bar.update_layout(title="Top-12 by WScore — Research preset", height=500, margin=dict(l=200,r=40,t=60,b=60))

fig_stack = go.Figure()
for k,m,_,_ in TERMS:
    fig_stack.add_trace(go.Bar(name=m, x=top["ConfigName"] if "ConfigName" in top else top.index.astype(str),
                               y=norm_cols[m].iloc[rank_idx[:12]]*PRESETS_N["Research"][k]))
fig_stack.update_layout(barmode="stack", title="Stacked metric contributions (Top-12)", height=500, legend=dict(orientation="h"))

best = top.iloc[0]
rvals = [float(norm_cols[m].iloc[rank_idx[0]]) for _,m,_,_ in TERMS]
fig_radar = go.Figure(go.Scatterpolar(r=rvals + [rvals[0]], theta=[m for _,m,_,_ in TERMS]+[TERMS[0][1]], fill="toself"))
fig_radar.update_layout(title="Radar — Best configuration", polar=dict(radialaxis=dict(range=[0,1])))

# convert to HTML fragments
fig_bar_div = fig_bar.to_html(full_html=False, include_plotlyjs='cdn')
fig_stack_div = fig_stack.to_html(full_html=False, include_plotlyjs=False)
fig_radar_div = fig_radar.to_html(full_html=False, include_plotlyjs=False)

# ---- HTML template with tab switcher ----
html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Pareto + Scores Dashboard</title>
<style>
body {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;margin:0;background:#fafafa;}}
header {{background:#0b3d91;color:#fff;padding:16px;}}
.tabs {{display:flex;border-bottom:2px solid #ddd;background:#f3f3f3;}}
.tabbtn {{flex:1;padding:12px;text-align:center;cursor:pointer;font-weight:600;}}
.tabbtn.active {{background:#fff;border-top:3px solid #0b3d91;}}
.tabcontent {{display:none;padding:16px;}}
.tabcontent.active {{display:block;}}
button.download {{padding:6px 12px;border:1px solid #ccc;border-radius:6px;background:#f7f7f7;cursor:pointer;}}
</style>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
</head>
<body>
<header><h2>Pareto vs Scores Interactive Dashboard</h2></header>
<div class='tabs'>
  <div class='tabbtn active' data-tab='pareto'>Pareto Dashboard</div>
  <div class='tabbtn' data-tab='scores'>Scores Analytics</div>
</div>

<div id='pareto' class='tabcontent active'>
  <iframe src='pareto_dashboard_presets_custom_constraints_score.html' style='width:100%;height:88vh;border:none;'></iframe>
</div>

<div id='scores' class='tabcontent'>
  <h3>Scores Analytics (Research preset demo)</h3>
  <div>{fig_bar_div}</div>
  <div>{fig_stack_div}</div>
  <div>{fig_radar_div}</div>
  <button class='download' onclick='downloadCSV()'>Download Scores CSV</button>
</div>

<script>
document.querySelectorAll('.tabbtn').forEach(btn=>{{
  btn.addEventListener('click',()=>{{
    document.querySelectorAll('.tabbtn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tabcontent').forEach(c=>c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }});
}});

function downloadCSV() {{
  const csv = `{top.to_csv(index=False)}`;
  const blob = new Blob([csv], {{type:'text/csv;charset=utf-8;'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'scores_research_top12.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""

out_html = os.path.join(OUT_DIR, "pareto_dashboard_presets_custom_constraints_score_radar.html")
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)

print("Integrated tabbed dashboard written to:", out_html)
