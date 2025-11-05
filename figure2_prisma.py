import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Example: load your original matrix data
# Replace with your real sensor-task data array
data = np.array([
    [2, 1, 2, 0, 0, 2, 0, 0],
    [1, 0, 3, 3, 1, 2, 0, 0],
    [0, 0, 4, 2, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 3],
    [0, 1, 2, 3, 0, 0, 0, 1]
], dtype=float)

sensors = [
    "Active acoustic (speaker+mic)",
    "Passive mic (acoustic)",
    "PPG (optical)",
    "IMU (acc/gyro)",
    "Electrodes / EEG",
    "Thermistor / Temp"
]
tasks = [
    "Authentication / ID",
    "Otology screening",
    "Heart / HR / HRV",
    "Respiration / Breathing",
    "Cough detection",
    "Activity recognition",
    "Fluid intake",
    "EEG / Electrophysiology"
]

fig, ax = plt.subplots(figsize=(9, 5))

# ✅ Use lighter IEEE-style colormap
#im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=3.5)
#im = ax.imshow(data, cmap="PuBuGns", vmin=0, vmax=3.5)
#im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=0.35)
#im = ax.imshow(data, aspect='auto', cmap='YlGnBu', vmin=0, vmax=0.35)
im = ax.imshow(data, aspect='auto', cmap='YlGnBu', vmin=-1, vmax=5)

# Axis labels
ax.set_xticks(np.arange(len(tasks)))
ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(sensors)))
ax.set_yticklabels(sensors, fontsize=9)

# Annotate values on top of cells
for i in range(len(sensors)):
    for j in range(len(tasks)):
        if data[i, j] > 0:
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=8, color="black")

# Add a colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Weighted Relevance", rotation=90)
cbar.ax.tick_params(labelsize=8)

# A few final touches
#ax.set_title("Figure 2 – Sensor–Task Relevance Matrix", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig("outputs/figs_prisma/Figure_2_Lighter_YlGnBu_01.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
