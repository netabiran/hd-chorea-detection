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

# Collect all valid windows from all subkjects
valid_windows_mask = (np.sum(chorea_walking >= 0, axis=1) >= 1)

# Concatenate all chorea samples from these windows (flattened)
valid_chorea_samples = chorea_walking[valid_windows_mask].flatten()
valid_chorea_samples = valid_chorea_samples[valid_chorea_samples >= 0]  # only valid samples

# Count samples per chorea class 0..4
chorea_classes = [0, 1, 2, 3, 4]
counts_per_class = [np.sum(valid_chorea_samples == c) for c in chorea_classes]

custom_colors = ["#2f9be9", "#2f9be9", '#2f9be9', '#2f9be9', '#2f9be9']  # blue, orange, green, red, purple
sns.barplot(x=chorea_classes, y=counts_per_class, palette=custom_colors)

plt.xlabel("Chorea Class", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Chorea Class Sample Counts in Valid Windows (≥1 samples) for all Subjects\n(Walking Only)", fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("hd-chorea-detection/chorea_class_counts_valid_windows_all_subjects_largerThan1.png")
plt.show()