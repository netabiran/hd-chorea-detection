import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np

# Paths
output_folder = "/home/netabiran/hd-chorea-detection/data_preprocessing/data/11TC/output_figures"
os.makedirs(output_folder, exist_ok=True)

df1 = pd.read_csv("/home/netabiran/hd-chorea-detection/data_preprocessing/data/11TC/walking_frames_with_chorea_11TC_NB.csv")
df2 = pd.read_csv("/home/netabiran/hd-chorea-detection/data_preprocessing/data/11TC/walking_frames_with_chorea_11TC_GH.csv")

# Get subject identifier from folder path
subject_id = os.path.basename(os.path.dirname(output_folder))

# Merge and clean
merged_df = df1.merge(df2, on="Frame", how="outer", suffixes=('_file1', '_file2'))
merged_df.fillna(-9, inplace=True)
merged_df["Chorea Label_file1"] = merged_df["Chorea Label_file1"].astype(int)
merged_df["Chorea Label_file2"] = merged_df["Chorea Label_file2"].astype(int)

# Filter valid comparisons (ignore -9)
valid_df = merged_df[(merged_df["Chorea Label_file1"] != -9) & (merged_df["Chorea Label_file2"] != -9)]

# Confusion matrix
cm = confusion_matrix(valid_df["Chorea Label_file1"], valid_df["Chorea Label_file2"], labels=[0, 1, 2, 3, 4])
kappa = cohen_kappa_score(valid_df["Chorea Label_file1"], valid_df["Chorea Label_file2"], weights='quadratic')

# Difference statistics
diff = np.abs(valid_df["Chorea Label_file1"] - valid_df["Chorea Label_file2"])
total = len(diff)
exact_matches = (diff == 0).sum()
mismatch_any = (diff > 0).sum()
mismatch_gt1 = (diff > 1).sum()
mismatch_gt2 = (diff > 2).sum()

# Percentage stats
pct_match = exact_matches / total * 100
pct_mismatch_any = mismatch_any / total * 100
pct_mismatch_gt1 = mismatch_gt1 / total * 100
pct_mismatch_gt2 = mismatch_gt2 / total * 100
mean_diff = diff.mean()
std_diff = diff.std()

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Chorea Label (File 2)")
plt.ylabel("Chorea Label (File 1)")
plt.title(
    f"Subject: {subject_id} | Kappa: {kappa:.2f}\n"
    f"Exact Match: {pct_match:.1f}% | >1 Diff: {pct_mismatch_gt1:.1f}% | "
    f">2 Diff: {pct_mismatch_gt2:.1f}% | Mean Diff: {mean_diff:.2f} ± {std_diff:.2f}"
)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix_with_kappa.png"))
plt.close()

# Segment visualization
frame_gaps = valid_df["Frame"].diff() > 1
segment_starts = np.where(frame_gaps)[0]
segment_starts = np.insert(segment_starts, 0, 0)
segments = [valid_df.iloc[segment_starts[i]:segment_starts[i+1]] for i in range(len(segment_starts)-1)]
segments.append(valid_df.iloc[segment_starts[-1]:])

segments_per_figure = 3
total_figures = int(np.ceil(len(segments) / segments_per_figure))

for fig_idx in range(total_figures):
    num_subplots = min(segments_per_figure, len(segments) - fig_idx * segments_per_figure)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots))

    if num_subplots == 1:
        axes = [axes]

    for i in range(num_subplots):
        segment_idx = fig_idx * segments_per_figure + i
        if segment_idx >= len(segments):
            break 

        segment = segments[segment_idx]
        ax = axes[i]

        matching = segment[segment["Chorea Label_file1"] == segment["Chorea Label_file2"]]
        mismatching = segment[segment["Chorea Label_file1"] != segment["Chorea Label_file2"]]

        ax.scatter(matching["Frame"], matching["Chorea Label_file1"], color='black', label="Match", alpha=0.6, s=10)
        ax.scatter(mismatching["Frame"], mismatching["Chorea Label_file1"], color='blue', label="File 1", alpha=0.6, s=10)
        ax.scatter(mismatching["Frame"], mismatching["Chorea Label_file2"], color='red', label="File 2", alpha=0.6, s=10)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Chorea Label")
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_title(f"Segment {segment_idx + 1} (Frames {segment['Frame'].min()}-{segment['Frame'].max()})")

    plt.tight_layout()
    output_path = os.path.join(output_folder, f"timeline_segment_{fig_idx+1}.png")
    plt.savefig(output_path)
    print(f"Saved figure: {output_path}")
    plt.close()
