# inject_explicit_forms.py  (fixed ordering for layer penalties before Pareto)
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV  = "csvs/tradeoffs_all_points_EXT.csv"
#OUT_DIR = "explicit_outputs"
OUT_DIR = "csvs"
#FIG_DIR = os.path.join(OUT_DIR, "figs")
FIG_DIR = os.path.join(OUT_DIR, "../outputs/figs_derived")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Load & backup ----------
df = pd.read_csv(IN_CSV).copy()
df.rename(columns={
    "Error":"Error_orig","Coverage":"Coverage_orig",
    "Latency_p95_s":"Latency_p95_s_orig","Energy_mWh_day":"Energy_mWh_day_orig"
}, inplace=True)

def col(df, name, default):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)

# ---------- Inputs / proxies ----------
df["f_samp"]   = col(df, "fs_factor", 1.0).astype(float).clip(lower=1e-6)
df["duty"]     = col(df, "duty", 0.5).astype(float).clip(0.0, 1.0)
df["motion"]   = df.get("motion", pd.Series(["low"]*len(df))).astype(str).str.lower()
df["Scenario"] = df.get("Scenario", pd.Series([""]*len(df))).astype(str).str.lower()
df["Model"]    = df.get("Model", pd.Series(["base"]*len(df))).astype(str).str.lower()
df["network"]  = df.get("network", pd.Series(["edge"]*len(df))).astype(str).str.lower()
if "Modality" not in df.columns:
    df["Modality"] = "all"

# ---------- Coefficients ----------
A_LOW, A_MED, A_HIGH = 0.00, 0.15, 0.35
SNR_TH, GAMMA_SNR    = 20.0, 0.03
BETA_FIT             = 0.30
FIT_CLIP             = (0.8, 1.1)
SENSE_FRAC, COMP_FRAC, NET_FRAC = 0.40, 0.35, 0.25
dE_IMU_mWh, dE_FIT_mWh = 8.0, 5.0
TAU_IMU_s            = 0.01
GAMMA_MAX, ALPHA_OFF = 0.30, 0.7
BATTERY_CAP_MWH      = 300.0

def a_L(m):
    m = m.lower()
    if "high" in m: return A_HIGH
    if "med"  in m: return A_MED
    return A_LOW

def snr_from_scenario(s):
    s = s.lower()
    if any(k in s for k in ["noisy","gym","street","bus","train","subway","crowd","factory"]): return 12.0
    if any(k in s for k in ["office","home","cafe","room","urban"]):                           return 22.0
    if any(k in s for k in ["quiet","lab","anechoic"]):                                       return 28.0
    return 20.0

def g_m(m):
    m = m.lower()
    if "tiny" in m: return 0.95
    if "small" in m:return 1.00
    if "base" in m: return 1.05
    if "large" in m or "xl" in m: return 1.10
    return 1.00

def chi_m(m):
    m = m.lower()
    if "tiny" in m: return 0.60
    if "small" in m:return 0.80
    if "base" in m: return 1.00
    if "large" in m or "xl" in m: return 1.40
    return 1.00

def phi_m(m):
    m = m.lower()
    if "tiny" in m: return 0.02
    if "small" in m:return 0.05
    if "base" in m: return 0.10
    if "large" in m or "xl" in m: return 0.20
    return 0.10

def eta_n(n):  return 1.10 if "edge+cloud" in n.lower() else 0.60
def lambda_n(n): return 0.20 if "edge+cloud" in n.lower() else 0.02
def gamma_cloud(n): return GAMMA_MAX * (1.0 ** ALPHA_OFF) if "edge+cloud" in n.lower() else 0.0

# ---------- Factors ----------
df["a_L"]       = df["motion"].apply(a_L)
df["MotionPen"] = 1.0 + df["a_L"] * 1.0
df["SNR_dB"]    = df["Scenario"].apply(snr_from_scenario)
df["NoisePen"]  = (1.0 + GAMMA_SNR * (SNR_TH - df["SNR_dB"]).clip(lower=0)).clip(lower=1.0)

I_high = (df["motion"]=="high").astype(float)
den_cov = (df["duty"] * (1.0 - 0.10 * 1.0 * I_high)).replace({0.0: np.nan})
p_fit_est = 0.5 + ((df["Coverage_orig"] / den_cov) - 1.0) / 0.05
df["p_fit"]   = p_fit_est.fillna(0.5).clip(0.0, 1.0)
df["FitGain"] = np.clip(1.0 - BETA_FIT * (df["p_fit"] - 0.5), *FIT_CLIP)

df["g_m"]       = df["Model"].apply(g_m)
df["chi_m"]     = df["Model"].apply(chi_m)
df["phi_m"]     = df["Model"].apply(phi_m)
df["eta_n"]     = df["network"].apply(eta_n)
df["lambda_n"]  = df["network"].apply(lambda_n)
df["gamma_cloud"]= df["network"].apply(gamma_cloud)
df["duty_term"] = 1.0 / np.maximum(0.5, 0.85 + 0.30*(df["duty"] - 0.5))

# ---------- Recompute explicit metrics ----------
df["eps_base"] = df["Error_orig"] * np.sqrt(df["f_samp"]) * df["g_m"] / (
    df["duty_term"] * df["MotionPen"] * df["NoisePen"] * df["FitGain"] * (1.0 - df["gamma_cloud"])
)
df["Error"] = df["eps_base"] / (np.sqrt(df["f_samp"]) * df["g_m"]) * \
              df["duty_term"] * df["MotionPen"] * df["NoisePen"] * df["FitGain"] * (1.0 - df["gamma_cloud"])

df["Coverage"] = np.clip(df["duty"] * (1.0 - 0.10 * I_high) * (1.0 + 0.05*(df["p_fit"] - 0.5)), 0.0, 1.0)

den_E = df["duty"] * (SENSE_FRAC*df["f_samp"] + COMP_FRAC*df["chi_m"] + NET_FRAC*df["eta_n"])
den_E = den_E.replace({0.0: np.nan})
E_base = (df["Energy_mWh_day_orig"] - dE_IMU_mWh - dE_FIT_mWh) / den_E
df["E_base"] = E_base.replace([np.inf, -np.inf], np.nan).fillna(E_base.median())

df["Energy_mWh_day"] = (df["E_base"] * df["duty"] *
                        (SENSE_FRAC*df["f_samp"] + COMP_FRAC*df["chi_m"] + NET_FRAC*df["eta_n"])
                        + dE_IMU_mWh + dE_FIT_mWh)

duty_lat = 1.0 + 0.10*(1.0 - df["duty"])
tau_base = (df["Latency_p95_s_orig"] / duty_lat) - (df["phi_m"] + df["lambda_n"] + TAU_IMU_s)
df["tau_base"] = tau_base
df["Latency_p95_s"] = (df["tau_base"] + df["phi_m"] + df["lambda_n"] + TAU_IMU_s) * duty_lat

# ---------- Derived ----------
df["BatteryDays"]  = BATTERY_CAP_MWH / df["Energy_mWh_day"].clip(lower=1e-9)
df["EDP"]          = df["Energy_mWh_day"] * df["Latency_p95_s"]
df["EEprod"]       = df["Energy_mWh_day"] * df["Error"]
def norm_error_pct(s):
    b = max(s.min(), 1e-12)
    return 100.0 * s / b
df["NormError_%"]  = df.groupby("Modality")["Error"].transform(norm_error_pct)

keys = ["Scenario","Modality","Model","fs_factor","duty","network"]
for k in keys:
    if k not in df.columns:
        df[k] = "" if k!="duty" else 0.5
lo = df[df["motion"]=="low"][keys+["Error"]].rename(columns={"Error":"Error_low"})
hi = df[df["motion"]=="high"][keys+["Error"]].rename(columns={"Error":"Error_high"})
upl = pd.merge(hi, lo, on=keys, how="left")
upl["StressUplift_%"] = 100.0*((upl["Error_high"] / upl["Error_low"].clip(lower=1e-9)) - 1.0)
df = pd.merge(df, upl[keys+["StressUplift_%"]], on=keys, how="left")
df["StressUplift_%"] = df["StressUplift_%"].fillna(0.0).clip(lower=0.0)
df["CloudBoost_pct"] = np.where(df["network"]=="edge+cloud", 100.0, 0.0)

# ---------- Layer penalties (moved BEFORE Pareto) ----------
def n01(s):
    s = s.astype(float); mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s*0.0

p_E   = n01(df["Energy_mWh_day"])
p_L   = n01(df["Latency_p95_s"])
p_Er  = n01(df["Error"])
p_Ci  = 1.0 - n01(df["Coverage"])
p_BDi = 1.0 - n01(df["BatteryDays"])
p_EEP = n01(df["EEprod"])
p_NE  = n01(df["NormError_%"])
p_SU  = n01(df["StressUplift_%"])
p_CB  = n01(df["CloudBoost_pct"])

df["L1_Device_pen"]     = 0.5*p_E + 0.3*p_Ci + 0.2*p_BDi
df["L2_Edge_pen"]       = 0.6*p_Er + 0.3*p_SU + 0.1*p_EEP
df["L3_Conn_pen"]       = 0.5*p_CB + 0.5*p_L
df["L5_Analytics_pen"]  = 0.7*p_NE + 0.3*p_CB

vals = df[["L1_Device_pen","L2_Edge_pen","L3_Conn_pen","L5_Analytics_pen"]].to_numpy()
labels = np.array(["L1_Device","L2_Edge","L3_Connectivity","L5_Analytics"])
df["PrimaryLayerConcern"] = labels[np.nanargmax(vals, axis=1)]

# ---------- Save updated dataset ----------
CSV_EXPLICIT = os.path.join(OUT_DIR, "tradeoffs_all_points_EXPLICIT.csv")
df.to_csv(CSV_EXPLICIT, index=False)

# ---------- Pareto utilities ----------
def pareto_front_mask(D: pd.DataFrame) -> pd.Series:
    X = np.column_stack([
        D["Error"].to_numpy(),
        -D["Coverage"].to_numpy(),
        D["Energy_mWh_day"].to_numpy(),
        D["Latency_p95_s"].to_numpy()
    ])
    n = X.shape[0]; dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]: continue
        le = (X <= X[i]).all(axis=1); lt = (X < X[i]).any(axis=1)
        dominated |= (le & lt); dominated[i] = False
    return ~dominated

pareto = df.loc[pareto_front_mask(df)].copy().reset_index(drop=True)
pareto.to_csv(os.path.join(OUT_DIR, "pareto_front_unconstrained.csv"), index=False)

# ---------- Sensitivity ----------
def sens4(P: pd.DataFrame, draws=400, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    def nmin(a):
        a = a.astype(float); mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx > mn else (a*0.0)
    M = np.column_stack([nmin(P["Error"].values),
                         nmin(-P["Coverage"].values),
                         nmin(P["Energy_mWh_day"].values),
                         nmin(P["Latency_p95_s"].values)])
    W = rng.dirichlet(np.ones(4), size=draws)
    picks = np.argmin(M @ W.T, axis=0)
    counts = pd.Series(picks).value_counts().sort_values(ascending=False)
    out = pd.DataFrame({"pareto_index": counts.index, "pick_count": counts.values})
    out["pick_pct"] = 100.0 * out["pick_count"] / draws
    out = out.merge(P.reset_index().rename(columns={"index":"pareto_index"}), on="pareto_index", how="left")
    return out

sens_uncon = sens4(pareto, 400, 11)
sens_uncon.to_csv(os.path.join(OUT_DIR, "pareto_weight_sensitivity_unconstrained.csv"), index=False)

# ---------- Constraints + knees ----------
COV_MIN, LAT_MAX, ENE_MAX = 0.65, 4.0, 300.0
feas = df[(df["Coverage"] >= COV_MIN) & (df["Latency_p95_s"] <= LAT_MAX) & (df["Energy_mWh_day"] <= ENE_MAX)].copy()
feas.to_csv(os.path.join(OUT_DIR, "feasible_constrained.csv"), index=False)

pareto_c = feas.loc[pareto_front_mask(feas)].copy().reset_index(drop=True)
pareto_c.to_csv(os.path.join(OUT_DIR, "pareto_front_constrained.csv"), index=False)
sens_con = sens4(pareto_c, 400, 22)
sens_con.to_csv(os.path.join(OUT_DIR, "pareto_weight_sensitivity_constrained.csv"), index=False)

def epsilon_knees(D: pd.DataFrame, steps=(0.05,0.10,0.15)):
    if D.empty: return pd.DataFrame()
    E_min, E_max = D["Energy_mWh_day"].min(), D["Energy_mWh_day"].max()
    L_min, L_max = D["Latency_p95_s"].min(), D["Latency_p95_s"].max()
    C_min, C_max = D["Coverage"].min(), D["Coverage"].max()
    rng_E, rng_L, rng_C = (E_max-E_min), (L_max-L_min), (C_max-C_min)
    rows = []
    for frac in steps:
        cand = D[(D["Energy_mWh_day"] <= E_min + frac*rng_E) &
                 (D["Latency_p95_s"] <= L_min + frac*rng_L) &
                 (D["Coverage"]      >= C_max - frac*rng_C)]
        if cand.empty: continue
        sel = cand.nsmallest(1, "Error").copy()
        sel["eps_fraction_used"] = frac
        rows.append(sel)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

knees = epsilon_knees(pareto_c, (0.05,0.10,0.15))
knees.to_csv(os.path.join(OUT_DIR, "knees_epsilon_constrained.csv"), index=False)

# ---------- Figures ----------
def n01p(s):
    s = s.astype(float); mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s*0.0

# (a) 3D Pareto cloud with layer markers and size = inverse penalty
fig1 = plt.figure(figsize=(10,7))
ax1  = fig1.add_subplot(111, projection="3d")
marker_map = {"L1_Device":"o","L2_Edge":"s","L3_Connectivity":"^","L5_Analytics":"D"}
# ensure penalty cols exist in pareto (they do now)
health = 1.0 - n01p(pareto["L1_Device_pen"] + pareto["L2_Edge_pen"] + pareto["L3_Conn_pen"] + pareto["L5_Analytics_pen"])
sizes = 40 + 200*health
for layer, mk in marker_map.items():
    sel = pareto["PrimaryLayerConcern"] == layer
    ax1.scatter(pareto.loc[sel,"Error"], pareto.loc[sel,"Coverage"], pareto.loc[sel,"Energy_mWh_day"],
                marker=mk, s=sizes[sel], alpha=0.85, label=layer)
ax1.set_xlabel("Error"); ax1.set_ylabel("Coverage"); ax1.set_zlabel("Energy (mWh/day)")
ax1.set_title("Pareto Front (unconstrained) — Explicit Forms Injected")
ax1.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_pareto3d_layer.png"), dpi=300, bbox_inches="tight")
plt.close(fig1)

# (b) Bubble trade: Error vs Energy, bubble size = Coverage
fig2 = plt.figure(figsize=(9,6)); ax2 = plt.gca()
ax2.scatter(pareto["Energy_mWh_day"], pareto["Error"], s=30 + 220*pareto["Coverage"].to_numpy(), alpha=0.8)
ax2.set_xlabel("Energy (mWh/day)"); ax2.set_ylabel("Error")
ax2.set_title("Pareto Projection: Error vs Energy (bubble size = Coverage)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_error_energy_bubble.png"), dpi=300, bbox_inches="tight")
plt.close(fig2)

# (c) Edge vs Edge+Cloud uplift
keys_cmp = ["Scenario","Modality","Model","fs_factor","duty","motion"]
edge = df[df["network"]=="edge"][keys_cmp+["Error","Energy_mWh_day"]].copy()
cloud= df[df["network"]=="edge+cloud"][keys_cmp+["Error","Energy_mWh_day"]].copy()
pair = pd.merge(edge, cloud, on=keys_cmp, suffixes=("_edge","_cloud"))
pair["dError_pct"]  = 100.0*(pair["Error_edge"] - pair["Error_cloud"])/pair["Error_edge"].replace(0,np.nan)
pair["dEnergy_mWh"] = pair["Energy_mWh_day_cloud"] - pair["Energy_mWh_day_edge"]
fig3 = plt.figure(figsize=(9,6)); ax3 = plt.gca()
ax3.scatter(pair["dEnergy_mWh"], pair["dError_pct"], alpha=0.85)
ax3.axhline(0.0, linestyle="--")
ax3.set_xlabel("ΔEnergy (cloud − edge), mWh/day"); ax3.set_ylabel("Error reduction with cloud (%)")
ax3.set_title("Edge vs Edge+Cloud: Error Uplift vs Energy Cost")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_edge_cloud_uplift.png"), dpi=300, bbox_inches="tight")
plt.close(fig3)

# (d) ε-knees overlay on constrained Pareto
fig4 = plt.figure(figsize=(9,6)); ax4 = plt.gca()
ax4.scatter(pareto_c["Energy_mWh_day"], pareto_c["Error"], alpha=0.7)
if not knees.empty:
    ax4.scatter(knees["Energy_mWh_day"], knees["Error"], marker="D", s=120, alpha=0.95)
ax4.set_xlabel("Energy (mWh/day)"); ax4.set_ylabel("Error")
ax4.set_title("Constrained Pareto: ε-knees (5/10/15%) on Error–Energy")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_knees_energy_error.png"), dpi=300, bbox_inches="tight")
plt.close(fig4)

# (e) Layer Health radar
health2 = df.copy()
health2["Accuracy"]       = 1.0 - n01(health2["Error"])
health2["CoverageHealth"] = n01(health2["Coverage"])
health2["Endurance"]      = 1.0 - n01(health2["Energy_mWh_day"])
health2["Responsiveness"] = 1.0 - n01(health2["Latency_p95_s"])
health2["Efficiency"]     = 1.0 - n01(health2["EEprod"])
axes = ["Accuracy","CoverageHealth","Endurance","Responsiveness","Efficiency"]
angles = np.linspace(0, 2*np.pi, len(axes), endpoint=False)
angles = np.concatenate([angles, angles[:1]])
rad = health2.groupby("PrimaryLayerConcern")[axes].mean()
fig5 = plt.figure(figsize=(8,8)); ax5 = plt.subplot(111, polar=True)
for grp, row in rad.iterrows():
    vals = np.concatenate([row.values, row.values[:1]])
    ax5.plot(angles, vals, linewidth=2, alpha=0.9, label=grp)
    ax5.fill(angles, vals, alpha=0.1)
ax5.set_xticks(angles[:-1]); ax5.set_xticklabels(axes); ax5.set_yticklabels([])
ax5.set_title("Layer Health Radar (outward = better)")
ax5.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_layer_health_radar.png"), dpi=300, bbox_inches="tight")
plt.close(fig5)

# (f) Parallel coordinates
cols_pc = ["Error","Coverage","Energy_mWh_day","Latency_p95_s","EEprod"]
pc = df[cols_pc].copy()
def n01_for_pc(s):
    s = s.astype(float); mn, mx = s.min(), s.max()
    return (s - mn)/(mx - mn) if mx>mn else s*0.0
pc_n = pd.DataFrame({c: n01_for_pc(pc[c]) for c in cols_pc}).sample(n=min(200, len(pc)), random_state=42)
fig6 = plt.figure(figsize=(11,6)); ax6 = plt.gca()
x = np.arange(len(cols_pc))
for _, row in pc_n.iterrows(): ax6.plot(x, row.values, alpha=0.15)
ax6.set_xticks(x); ax6.set_xticklabels(cols_pc); ax6.set_ylim(0,1)
ax6.set_title("Parallel Coordinates (normalized) — sample of solutions")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_parallel_coordinates.png"), dpi=300, bbox_inches="tight")
plt.close(fig6)

# ---------- Index ----------
idx = pd.DataFrame([
    ("Explicit dataset CSV",                   os.path.join(OUT_DIR, "tradeoffs_all_points_EXPLICIT.csv")),
    ("Pareto (unconstrained) CSV",             os.path.join(OUT_DIR, "pareto_front_unconstrained.csv")),
    ("Pareto sensitivity (unconstrained) CSV", os.path.join(OUT_DIR, "pareto_weight_sensitivity_unconstrained.csv")),
    ("Feasible (constrained) CSV",             os.path.join(OUT_DIR, "feasible_constrained.csv")),
    ("Pareto (constrained) CSV",               os.path.join(OUT_DIR, "pareto_front_constrained.csv")),
    ("Pareto sensitivity (constrained) CSV",   os.path.join(OUT_DIR, "pareto_weight_sensitivity_constrained.csv")),
    ("ε-knees (constrained) CSV",              os.path.join(OUT_DIR, "knees_epsilon_constrained.csv")),
    ("Fig1 Pareto 3D",                         os.path.join(FIG_DIR, "fig1_pareto3d_layer.png")),
    ("Fig2 Error–Energy bubble",               os.path.join(FIG_DIR, "fig2_error_energy_bubble.png")),
    ("Fig3 Edge vs Cloud uplift",              os.path.join(FIG_DIR, "fig3_edge_cloud_uplift.png")),
    ("Fig4 ε-knees overlay",                   os.path.join(FIG_DIR, "fig4_knees_energy_error.png")),
    ("Fig5 Layer Health radar",                os.path.join(FIG_DIR, "fig5_layer_health_radar.png")),
    ("Fig6 Parallel coordinates",              os.path.join(FIG_DIR, "fig6_parallel_coordinates.png")),
], columns=["artifact","path"])
idx.to_csv(os.path.join(OUT_DIR, "INDEX.csv"), index=False)

print("Done. See:", os.path.join(OUT_DIR, "INDEX.csv"))
