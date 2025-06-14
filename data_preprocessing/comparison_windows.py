import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Parameters
window_size = 30  # Number of frames to look back and forward for matching
output_folder = "/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/output_figures"
os.makedirs(output_folder, exist_ok=True)

# Load data
df1 = pd.read_csv("/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/walking_frames_with_chorea_10TC_NB.csv")
df2 = pd.read_csv("/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/walking_frames_with_chorea_10TC_GH.csv")

subject_id = os.path.basename(os.path.dirname(output_folder))

# Merge
merged_df = df1.merge(df2, on="Frame", how="outer", suffixes=('_file1', '_file2'))
merged_df.fillna(-9, inplace=True)
merged_df["Chorea Label_file1"] = merged_df["Chorea Label_file1"].astype(int)
merged_df["Chorea Label_file2"] = merged_df["Chorea Label_file2"].astype(int)

# Only consider frames where both raters labeled something
valid_df = merged_df[(merged_df["Chorea Label_file1"] != -9) & (merged_df["Chorea Label_file2"] != -9)].copy()

# --- Windowed matching based on labels 0-4 ---
labels_file2_series = df2.set_index('Frame')["Chorea Label"]

matches = []
diffs = []
true_labels = []
predicted_labels = []

for idx, row in valid_df.iterrows():
    frame = row['Frame']
    label1 = row['Chorea Label_file1']

    # Define window
    start_frame = frame - window_size
    end_frame = frame + window_size

    # Clip to available frames
    window_labels = labels_file2_series.loc[(labels_file2_series.index >= start_frame) & (labels_file2_series.index <= end_frame)]

    if label1 in window_labels.values:
        matches.append(1)  # Match found within window
        diffs.append(0)
        predicted_label = label1
    else:
        matches.append(0)  # No match
        if len(window_labels) > 0:
            diffs.append(np.min(np.abs(window_labels.values - label1)))
            predicted_label = pd.Series(window_labels.values).mode()[0]  # Most frequent label
        else:
            diffs.append(np.nan)
            predicted_label = -1  # Should not happen if windows are correctly set

    true_labels.append(label1)
    predicted_labels.append(predicted_label)

valid_df["Match"] = matches
valid_df["Windowed_Diff"] = diffs

# Remove cases where predicted_label == -1 (if any)
valid_df = valid_df[valid_df["Match"].notna()]

# --- Statistics ---
total = len(valid_df)
match_count = valid_df["Match"].sum()
mismatch_count = total - match_count

pct_match = match_count / total * 100
pct_mismatch = mismatch_count / total * 100
mean_diff = np.nanmean(valid_df["Windowed_Diff"])
std_diff = np.nanstd(valid_df["Windowed_Diff"])

# --- Kappa calculation on windowed label prediction ---
kappa_windowed = cohen_kappa_score(true_labels, predicted_labels, weights='quadratic')

# --- Confusion Matrix ---
cm_windowed = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_windowed, annot=True, fmt='d', cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Chorea Label (Windowed File 2)")
plt.ylabel("Chorea Label (File 1)")
plt.title(
    f"Subject: {subject_id}\n"
    f"Windowed Kappa: {kappa_windowed:.2f}"
)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "windowed_chorea_confusion_matrix_counts.png"))
plt.close()

# --- Normalized Confusion Matrix ---
cm_windowed_norm = cm_windowed.astype('float') / cm_windowed.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_windowed_norm, annot=True, fmt='.2f', cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Chorea Label (Windowed File 2)")
plt.ylabel("Chorea Label (File 1)")
plt.title(
    f"Subject: {subject_id}\n"
    f"Normalized Windowed Confusion Matrix"
)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "windowed_chorea_confusion_matrix_normalized.png"))
plt.close()

print(f"Saved confusion matrices at {output_folder}")

# --- Plot histogram of diffs ---
plt.figure(figsize=(8, 5))
sns.histplot(valid_df["Windowed_Diff"].dropna(), bins=np.arange(-0.5, 5.5, 1), kde=False, color='purple', edgecolor='black')
plt.xlabel("Absolute Difference Between Raters (Windowed)")
plt.ylabel("Number of Frames")
plt.title(f"Subject: {subject_id} | Absolute Differences (Windowed)")
plt.xticks([0, 1, 2, 3, 4, 5])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "absolute_difference_distribution_windowed.png"))
plt.close()

# --- Pie chart of match vs mismatch ---
plt.figure(figsize=(6, 6))
plt.pie([match_count, mismatch_count], labels=["Match", "Mismatch"], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=140)
plt.title(f"Subject: {subject_id} | Match vs Mismatch Rate (Windowed)")
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "match_vs_mismatch_pie_windowed.png"))
plt.close()

# --- Timeline Segment Visualization ---
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

        matching = segment[segment["Match"] == 1]
        mismatching = segment[segment["Match"] == 0]

        ax.scatter(matching["Frame"], matching["Chorea Label_file1"], color='black', label="Match", alpha=0.6, s=10)
        ax.scatter(mismatching["Frame"], mismatching["Chorea Label_file1"], color='blue', label="Mismatch", alpha=0.6, s=10)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Chorea Label (File 1)")
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_title(f"Segment {segment_idx + 1} (Frames {segment['Frame'].min()}-{segment['Frame'].max()})")

    # plt.tight_layout()
    # output_path = os.path.join(output_folder, f"timeline_segment_windowed_{fig_idx+1}.png")
    # plt.savefig(output_path)
    # print(f"Saved figure: {output_path}")
    # plt.close()
