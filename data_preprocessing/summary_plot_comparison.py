import matplotlib.pyplot as plt
import numpy as np
import os

output_folder = "/home/netabiran/hd_gait_detection_with_SSL/data_preprocessing"

# Subject identifiers
subjects = [f"Subject {i}" for i in [5,6,7,8,9,10]]

# Kappa scores
framewise_kappa = [0.27, 0.89, 0.94, 0.72, 0.39, 0.43]
windowed_kappa = [0.60, 1.00, 0.97, 0.80, 0.60, 0.63]
old_windowed_kappa = [0.34, 0.00, -0.02, 0.41, 0.07, 0.39]

# Plot configuration
x = np.arange(len(subjects))  # the label locations
width = 0.25  # width of the bars

# Red palette colors
color_framewise = '#ff9999'       # light red
color_windowed = '#ff4d4d'        # medium red
color_old_windowed = '#b30000'    # dark red

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, framewise_kappa, width, label='Framewise', color=color_framewise)
rects2 = ax.bar(x, windowed_kappa, width, label='Windowed (New Labels)', color=color_windowed)
rects3 = ax.bar(x + width, old_windowed_kappa, width, label='Windowed (Old Labels)', color=color_old_windowed)

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel("Cohen's Kappa")
ax.set_xlabel("Subject")
ax.set_title("Comparison of Cohen's Kappa Across Methods")
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(-0.1, 1.1)
ax.legend()

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_folder, "summary_kappa_comparison.png"), dpi=300)
plt.show()
