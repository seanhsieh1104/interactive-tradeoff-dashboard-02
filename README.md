# Earable IoT Pareto Dashboard (Research Companion Code)

This repository provides the **companion source code** for the manuscript:

> **Smart Earables for Continuous Health Monitoring: A Simulation-Driven Trade-off Framework**  
> *Shang-Chen Hsieh, Sung-Huai Hsieh 2025.*

It reproduces the metric derivation pipeline, multi-objective Pareto dashboards, and figure generation process used in the study.  
All scripts are designed for transparent, reproducible, and FAIR-aligned engineering analysis of smart earable systems.

---

## 🧠 Overview

The repository implements a **simulation-driven workflow** that connects raw performance metrics
(Error, Coverage, Latency, Energy) to derived indicators (BatteryDays, EDP, EEP, NormError, StressUplift, CloudBoost)  
and visualizes trade-offs through interactive 3-D Pareto dashboards and analytical figures.

**Workflow summary:**

1. **Metric Derivation** – Normalize, scale, and combine multi-objective data into a comparable score table.  
2. **Dashboard Generation** – Render interactive 3-D Pareto surfaces with ε-knee discovery, constraint filtering, and human-in-the-loop reweighting.  
3. **Figure Reproduction** – Automatically capture petal diagrams, heatmaps, and Pareto screenshots for publication-quality figures.

---

## 📁 Repository Structure

```text
.
├── csvs/
│   └── tradeoffs_all_points_EXT.csv        # baseline simulation data (input)
├── html/                                   # exported interactive dashboards
├── outputs/
│   ├── figs_derived/                       # derived metrics plots
│   ├── figs_petal_heatmap/                 # Fig. 6–7 visuals
│   └── figs_dashboard_rank/                # Fig. 4–5 visuals
├── derived_metrics.py                      # Step 1: derive metrics & Pareto sets
├── dashboard_01.py                         # Step 2: base Pareto dashboard
├── dashboard_02.py                         # Step 2b: dashboard + analytics tab
├── figure67_petal_heatmap.py               # Step 3a: petal & heatmap figures
├── figure45_capture.py                     # Step 3b: capture dashboard figures
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Dependencies include `numpy`, `pandas`, `plotly`, `matplotlib`, `selenium`, and `Pillow`.
Python 3.9–3.12 is supported.

---

## 🚀 Complete Workflow

### 🧩 Step 1 — Derive Metrics and Pareto Fronts

```bash
python derived_metrics.py
```

**Purpose:**  
Generates normalized and derived performance metrics and identifies Pareto-optimal configurations.

**Outputs:**
- `csvs/tradeoffs_all_points_EXPLICIT.csv`
- `csvs/pareto_front_constrained.csv`
- `csvs/pareto_front_unconstrained.csv`
- `outputs/figs_derived/` (summary plots)

These CSVs serve as the foundation for all later dashboards and figures.

---

### 📊 Step 2 — Build Interactive Dashboards

#### a) Base Pareto Dashboard
```bash
python dashboard_01.py
```
Creates:
```
html/pareto_dashboard_presets_custom_constraints_score.html
```

#### b) Extended Dashboard with “Scores Analytics”
```bash
python dashboard_02.py
```
Creates:
```
html/pareto_dashboard_presets_custom_constraints_score_radar.html
```

**Features**
- Two synchronized 3-D Pareto plots  
- ε-knee discovery and constraint filtering  
- Weight sliders for multi-objective rebalancing  
- Layer coloring by IoT level (device, edge, cloud)  
- Radar and contribution charts for top-ranked designs  

Open the generated HTML locally (`file://...`) — no server required.

---

### 🌸 Step 3 a — Generate Petal & Heatmap Figures (Fig. 6–7)

```bash
python figure67_petal_heatmap.py
```

Outputs:
```
outputs/figs_petal_heatmap/Fig6_Petal_*.png
outputs/figs_petal_heatmap/Fig7_Combined_Heatmap_AllSixScenarios.png
```

Each petal diagram encodes 10-metric composition and metadata  
(WScore, IoT layer, design focus) for six representative scenarios.

---

### 📸 Step 3 b — Capture Dashboard Figures (Fig. 4–5)

Requires **Google Chrome/Chromium** and **ChromeDriver** (matching version) in your PATH.

```bash
python figure45_capture.py
```

This script launches Selenium, loads the dashboards, and saves
publication-consistent screenshots to:

```
outputs/figs_dashboard_rank/
├── Fig4_Pareto_iframe.png
├── Fig5_Score_Ranking.png
├── Fig5a_Contribution_Stack.png
└── Fig5b_Radar_Profile.png
```

---

## 🧩 Data Notes

- Column names follow those defined in the manuscript:  
  `Error`, `Coverage`, `Latency_p95_s`, `Energy_mWh_day`, `BatteryDays`, `EDP`, `EEprod`, `NormError_%`, `StressUplift_%`, `CloudBoost_pct`, etc.  
- If your dataset differs slightly, the scripts include internal alias matching (`pick(...)`) to maintain compatibility.

---

## 🧾 Citation

If you use this repository in your work, please cite:

---

## 🔐 License

This repository is distributed under a **“For Research Use Only”** license.

> © 2025 Shang-Chen Hsieh.  
> The Software may be used, copied, and modified **solely for academic and non-commercial research**.  
> Commercial use is prohibited without prior written permission.  
> See [LICENSE](LICENSE) for full terms.

---

## 🙌 Acknowledgment

This open-data, open-code release aims to foster transparent and reproducible
design evaluation for **Smart Earables for Continuous Health Monitoring: A Simulation-Driven Trade-off Framework**.  
By combining data, scoring logic, and visuals in a single executable artifact,
it bridges quantitative analysis with implementable system design choices.
