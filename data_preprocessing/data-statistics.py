import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('data_ready/windows_input_to_multiclass_model.npz', allow_pickle=True)

# Constants
WALKING_LABEL = 1

# Load data arrays
chorea = data['win_chorea_all_sub']  # shape (N, window_size)
activity = data['win_labels_all_sub']  # shape (N, window_size)
subjects = data['win_subjects'].flatten()  # shape (N,)

# Replace chorea label -9 with -1 (invalid)
chorea[chorea == -9] = -1

# Mask windows that contain walking only:
# We'll consider a window valid only if it contains *any* walking sample (label=1)
walking_mask = np.any(activity == WALKING_LABEL, axis=1)

# Apply mask to chorea and subjects
chorea_walking = chorea[walking_mask]
subjects_walking = subjects[walking_mask]

# ----- Counting valid windows per subject -----
unique_subjects = np.unique(subjects_walking)
subject_valid_window_counts = []

for subj in unique_subjects:
    subj_mask = (subjects_walking == subj)
    subj_chorea_windows = chorea_walking[subj_mask]  # (num_windows, window_size)
    
    # Count windows with at least 100 valid chorea samples (>=0)
    count_valid = np.sum(np.sum(subj_chorea_windows >= 0, axis=1) >= 1)
    subject_valid_window_counts.append((subj, count_valid))

# Sort by count descending
subject_valid_window_counts.sort(key=lambda x: x[1], reverse=True)
subj_names, counts = zip(*subject_valid_window_counts)

# Colors: red for non-CO, blue for CO
colors = ['#FF9999' if 'CO' not in s else '#99CCFF' for s in subj_names]

# Total counts by group
total_co = sum(c for s, c in subject_valid_window_counts if 'CO' in s)
total_non_co = sum(c for s, c in subject_valid_window_counts if 'CO' not in s)

# ----- Plot all subjects -----
plt.figure(figsize=(14, 6))
bars = plt.bar(subj_names, counts, color=colors)
plt.xticks(rotation=90)
plt.xlabel('Subject')
plt.ylabel('Valid Chorea Windows (≥ 1 valid samples)')
plt.title('Number of Valid Chorea Windows per Subject (Walking Only)')
plt.grid(axis='y')

# Text box with totals
textstr = f'Total windows\nCO subjects: {total_co}\nNon-CO subjects: {total_non_co}'
plt.gcf().text(0.85, 0.75, textstr, fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

plt.tight_layout()
plt.savefig("hd-chorea-detection/number_of_valid_windows_per_subject_largerThan1.png")
plt.show()

# ----- Plot only non-CO subjects -----
non_co_subjects = [s for s in subj_names if 'CO' not in s]
non_co_counts = [c for s, c in subject_valid_window_counts if 'CO' not in s]

plt.figure(figsize=(12, 5))
plt.bar(non_co_subjects, non_co_counts, color='#FF9999')  # light red
plt.xticks(rotation=90)
plt.title('Valid Chorea Windows per Subject (Walking Only) - Non-CO Subjects')
plt.xlabel('Subject')
plt.ylabel('Valid Chorea Windows (≥ 1 valid samples)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("hd-chorea-detection/valid_windows_non_CO_subjects_largerThan1.png")
plt.show()

# ----- Plot only CO subjects -----
co_subjects = [s for s in subj_names if 'CO' in s]
co_counts = [c for s, c in subject_valid_window_counts if 'CO' in s]

plt.figure(figsize=(12, 5))
plt.bar(co_subjects, co_counts, color='#99CCFF')  # light blue
plt.xticks(rotation=90)
plt.title('Valid Chorea Windows per Subject (Walking Only) - CO Subjects')
plt.xlabel('Subject')
plt.ylabel('Valid Chorea Windows (≥ 1 valid samples)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("hd-chorea-detection/valid_windows_CO_subjects_largerThan1.png")
plt.show()

# ----- Heatmap of chorea class counts per subject (walking only) -----
classes = [0, 1, 2, 3, 4]
subject_ids = unique_subjects
counts = np.zeros((len(subject_ids), len(classes)))

for i, sid in enumerate(subject_ids):
    subj_mask = (subjects_walking == sid)
    subj_chorea = chorea_walking[subj_mask].flatten()
    subj_chorea = subj_chorea[subj_chorea >= 0]
    for j, c in enumerate(classes):
        counts[i, j] = np.sum(subj_chorea == c)

plt.figure(figsize=(12, 6))
sns.heatmap(counts, xticklabels=classes, yticklabels=subject_ids, annot=True, fmt='g', cmap="YlGnBu")
plt.title("Chorea Class Count Per Subject (Walking Only)")
plt.xlabel("Chorea Class")
plt.ylabel("Subject ID")
plt.tight_layout()
plt.savefig("hd-chorea-detection/Chorea_class_count_per_subject_walking_only_largerThan1.png")
plt.show()
