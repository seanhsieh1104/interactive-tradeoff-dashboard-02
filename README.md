# Earable IoT Pareto Dashboard (Research Companion Code)

This repository provides the **companion source code** for the manuscript:

> **Smart Earables for Continuous Health Monitoring: A Simulation-Driven Trade-off Framework**  
> *Shang-Chen Hsieh, Sung-Huai Hsieh 2025.*

The introduction video for this paper is available at: 
https://drive.google.com/file/d/1xkywt-Ax6O9b7-ZBjJ9w9cOVqxZtrdSf/view?usp=sharing

It reproduces the metric derivation pipeline, multi-objective Pareto dashboards, and figure generation process used in the study.  
All scripts are designed for transparent, reproducible, and FAIR-aligned engineering analysis of smart earable systems.

---

## 🧠 Overview

The repository implements a **simulation-driven workflow** that connects raw performance metrics
(Error, Coverage, Latency, Energy) to derived indicators (BatteryDays, EDP, EEP, NormError, StressUplift, CloudBoost)  
and visualizes trade-offs through interactive 3-D Pareto dashboards and analytical figures.

**Workflow summary:**

1. **Metric Derivation** – Normalize, scale, and combine multi-objective data into a comparable score table. The 
   metrics used in this paper are listed below.
      1.1. Definition: Sensing Error:   
        For modality i, the expected error (expressed as a percentage) is modeled as a multiplicative product of a 
        sampling term, model-tier factor, duty factor, motion penalty, noise penalty, and fit-quality factor. Error decreases with higher sampling and stronger models, increases under motion, noise, or poor fit, and can be further reduced by an optional cloud-assist term.

      1.2. Definition: Coverage 
        For modality i, Coverage is the effective fraction of valid measurements in a day, bounded in [0,1]. It scales with duty factor, degrades with motion, and improves with better fit quality; optional uptime and link-loss terms can be included when networking affects validity.

      1.3. Definition: Daily Energy. 
        Total daily energy per modality is modeled as the sum of sensing, compute, and networking contributions, plus IMU/fit-detection overhead and a small random variability term. The model captures trade-offs across these stages: energy increases with higher sampling rates, larger inference models, and cloud transport, and decreases with reduced duty cycle and greater on-device processing. The unit of Daily Energy is milliwatt-hour per day (mWh/day).

      1.4. Definition: Latency p95
        The 95th-percentile latency, with units of seconds (s), is modeled as baseline pipeline time plus model compute time and network transport time, with additional duty-cycle spin-up overhead and a multiplicative jitter term. Latency increases at low duty factors, with heavier models, and with larger network delays, reflecting degraded responsiveness.
      
      1.5. Derived metircs
        From the core models in Definitions (1)-(4), we define six derived indicators that summarize day-scale endurance, energy-latency coupling, energy-error trade-offs, error comparability, stress sensitivity, and offload benefit. The derived metrics are listed as below:

      1.6 COMPOSITE SCORE
        This score defines a composite WScore as a normalized weighted sum of min-max normalized metrics; lower values indicate better overall performance under the chosen preset. The metric set spans IoT layers (device, edge, connectivity, data, analytics, and operations), so cross-layer effects enter through which metrics are included and how they are weighted. Because all terms are min–max normalized within the current configuration grid, WScore expresses relative efficiency within this dataset; expanding the grid or adding new modalities will rescale the scores.      
 
2. **Dashboard Generation** – Render interactive 3-D Pareto surfaces with ε-knee discovery, constraint filtering, and human-in-the-loop reweighting. The screenshot of the dashboard is shown below.


3. **Figure Reproduction** – Automatically capture petal diagrams, heatmaps, and Pareto screenshots for publication-quality figures.

    3.1. Metric-Contribution Composition

    3.2. Radar Profile of the Optimal Configuration

    3.3. Metric-Contribution Heatmap Summary

    3.4. SIX REPRESENTATIVE
        (1) Battery-Dominant

        (2) Latency-Sensitive

        (3) Cloud-Assisted

        (4) Stress-Robust 

        (5) Throughput-Optimized

        (6) Balanced-Clinical 

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
https://github.com/seanhsieh1104/interactive-tradeoff-dashboard-02.git
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
