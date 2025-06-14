import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('data_ready/windows_input_to_multiclass_model.npz', allow_pickle=True)

WALKING_LABEL = 1

chorea = data['win_chorea_all_sub']
activity = data['win_labels_all_sub']
subjects = data['win_subjects'].flatten()

chorea[chorea == -9] = -1
walking_mask = np.any(activity == WALKING_LABEL, axis=1)

chorea_walking = chorea[walking_mask]
subjects_walking = subjects[walking_mask]

unique_subjects = np.unique(subjects_walking)

# Count valid windows per subject as before
subject_valid_window_counts = []
for subj in unique_subjects:
    subj_mask = (subjects_walking == subj)
    subj_chorea_windows = chorea_walking[subj_mask]
    count_valid = np.sum(np.sum(subj_chorea_windows >= 0, axis=1) >= 100)
    subject_valid_window_counts.append((subj, count_valid))
subject_valid_window_counts.sort(key=lambda x: x[1], reverse=True)

subj_names, counts = zip(*subject_valid_window_counts)

# Filter to only non-CO subjects with at least one valid window
non_co_subjects = [s for s in subj_names if 'CO' not in s]
valid_non_co_subjects = [s for s in non_co_subjects if dict(subject_valid_window_counts)[s] > 0]

# Collect all valid windows from those non-CO subjects
valid_mask = np.array([ (subj in valid_non_co_subjects) for subj in subjects_walking ])
valid_windows_mask = (np.sum(chorea_walking >= 0, axis=1) >= 1)
final_mask = valid_mask & valid_windows_mask

# Concatenate all chorea samples from these windows (flattened)
valid_chorea_samples = chorea_walking[final_mask].flatten()
valid_chorea_samples = valid_chorea_samples[valid_chorea_samples >= 0]  # only valid samples

# Count samples per chorea class 0..4
chorea_classes = [0, 1, 2, 3, 4]
counts_per_class = [np.sum(valid_chorea_samples == c) for c in chorea_classes]

custom_colors = ["#2f9be9", "#2f9be9", '#2f9be9', '#2f9be9', '#2f9be9']  # blue, orange, green, red, purple
sns.barplot(x=chorea_classes, y=counts_per_class, palette=custom_colors)

plt.xlabel("Chorea Class", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Chorea Class Sample Counts in Valid Windows (≥1 samples) for Non-CO Subjects\n(Walking Only)", fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("hd-chorea-detection/chorea_class_counts_valid_windows_non_CO_largerThan1.png")
plt.show()