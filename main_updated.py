import preprocessing
import numpy as np
import os
import torch
from sslmodel import get_sslnet
from validate_model import validate_model
import torch.nn.functional as F
from model_evaluation import auc_from_probs, compute_auc_from_outputs
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupKFold
from train_on_all_subjects import train_on_all_subjects_function

RAW_DATA_AND_LABELS_DIR = '/home/netabiran/data_ready/hd_dataset/lab_geneactive/synced_labeled_data'

preprocessing_mode = False
use_features = True
use_ordinal_loss = True  # True: ordinal loss (K-1 outputs for K classes), False: masked cross-entropy (K outputs)
use_class_weights = False  # True: use class weights to handle imbalanced data, False: no weighting
use_combined_labels = True  # True: combine labels (0->0, 1,2->1, 3,4->2), False: keep original 5 classes (0-4)
use_random_split = False  # True: random GroupKFold split, False: use predefined fold_to_val_subjects
use_leave_one_out = True  # True: Leave-One-Subject-Out CV (overrides use_random_split and n_folds)
train_on_all_subjects = False  # True: train on all subjects (no CV), False: use cross-validation
random_seed = 42  # Seed for reproducibility when using random split
n_folds = 5  # Number of folds for random split (only used when use_random_split=True and use_leave_one_out=False)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
max_epochs = 20  # Maximum number of training epochs
use_early_stopping = False  # True: enable early stopping, False: train for full max_epochs
early_stopping_patience = 15  # Stop if no improvement for this many epochs (only used if use_early_stopping=True)
min_epochs = 10  # Minimum epochs before early stopping can trigger (only used if use_early_stopping=True)
learning_rate = 1e-3  # Initial learning rate

# ============================================================
# WEIGHTS & BIASES (WANDB) CONFIGURATION
# ============================================================
use_wandb = False  # True: enable wandb logging, False: disable
wandb_project = "hd-chorea-detection"  # Wandb project name
wandb_entity = None  # Wandb entity/team (None for personal account)
wandb_run_name = "trial_new_script"  # Custom run name (None for auto-generated)
wandb_tags = []  # List of tags for this run (e.g., ["experiment1", "baseline"])
wandb_api_key = "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"  # Wandb API key (None to use default from wandb login or environment)
wandb_offline = False  # True: run in offline mode, False: sync to wandb servers

# ============================================================
# VALIDATION-ONLY MODE CONFIGURATION
# ============================================================
validation_only_mode = False  # True: only run validation on specified subjects using saved model
validation_model_path = "/home/netabiran/hd-chorea-detection/figures_output/new_labels_combined_best_model/saved_models/model_fold4.pth"  # Path to saved model
validation_subjects = ['IW12TC', 'IW13TC', 'IW14TC', 'IW15TC']  # List of subjects to validate on

curr_dir = os.getcwd()

PROCESSED_DATA_DIR = os.path.join(curr_dir, 'data_ready')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(curr_dir, 'model_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SSL_OUTPUT_DIR = os.path.join(curr_dir, 'ssl_outputs')

VIZUALIZE_DIR = "/home/netabiran/hd-chorea-detection/figures_output/final_results/trial_new_script_1702/"
os.makedirs(VIZUALIZE_DIR, exist_ok=True)

SRC_SAMPLE_RATE = int(100)
STD_THRESH = 0.1
WINDOW_SIZE = int(30*10)
WINDOW_OVERLAP = int(30*5)


# ============================================================
# DATASET AND LOSS CLASSES
# ============================================================

class ChoreaDataset(Dataset):
    def __init__(self, X, y, mask, features=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        if features is not None:
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.features is not None:
            return self.X[idx], self.y[idx], self.mask[idx], self.features[idx]
        else:
            return self.X[idx], self.y[idx], self.mask[idx]


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights)

    def forward(self, input, target, mask):
        loss = self.ce(input, target)
        masked_loss = loss * mask
        return masked_loss.sum() / (mask.sum() + 1e-6)


class CumulativeOrdinalLoss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.class_weights = class_weights

    def forward(self, logits, labels, mask):
        B, K, T = logits.shape
        labels_bin = torch.zeros((B, K, T), device=logits.device)
        for k in range(K):
            labels_bin[:, k, :] = (labels >= (k + 1)).float()
        loss = self.bce(logits, labels_bin)
        loss = loss * mask.unsqueeze(1)

        # Apply class weights if provided
        if self.class_weights is not None:
            sample_weights = torch.ones_like(mask)
            for c in range(len(self.class_weights)):
                sample_weights[labels == c] = self.class_weights[c]
            loss = loss * sample_weights.unsqueeze(1)

        return loss.sum() / (mask.sum() * K + 1e-8)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(acc_window, fs=30):
    """
    Extract time-domain and frequency-domain features from a single accelerometer window.

    acc_window: np.array shape [3, WINDOW_SIZE]
    fs: sampling frequency (Hz)
    Returns: 1D np.array of features
    """
    features = []

    # --- Time domain ---
    for axis in range(acc_window.shape[0]):
        x = acc_window[axis]

        # mean, std
        features.append(x.mean())
        features.append(x.std())

        # velocity and jerk
        v = np.diff(x, n=1, prepend=x[0])
        j = np.diff(v, n=1, prepend=v[0])
        features.append(v.mean())
        features.append(v.std())
        features.append(j.mean())
        features.append(j.std())

    # --- Frequency domain ---
    for axis in range(acc_window.shape[0]):
        x = acc_window[axis]
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1/fs)
        psd = np.abs(X)**2

        # power in bands
        bands = [(0.5, 3), (3, 7), (7, 12)]
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            band_power = psd[mask].mean() if np.any(mask) else 0
            features.append(band_power)

    return np.array(features)


def compute_all_window_features(win_acc_data):
    """
    Apply feature extraction to all windows.

    Returns window_features array if use_features is True, else None.
    """
    if not use_features:
        return None

    feature_list = [extract_features(win_acc_data[i]) for i in range(win_acc_data.shape[0])]
    window_features = np.stack(feature_list)  # shape: [num_windows, num_features]
    print("Window features shape:", window_features.shape)
    return window_features


# ============================================================
# WANDB INITIALIZATION
# ============================================================

def _resolve_wandb_api_key():
    """Try to find the wandb API key from multiple sources."""
    api_key = wandb_api_key
    if api_key is not None:
        return api_key

    # Check environment variable
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key is not None:
        return api_key

    # Try .netrc file
    try:
        import netrc
        from pathlib import Path
        netrc_path = Path.home() / '.netrc'
        if netrc_path.exists():
            nrc = netrc.netrc(netrc_path)
            for host in ['api.wandb.ai', 'wandb.ai']:
                try:
                    _, _, password = nrc.authenticators(host)
                    if password:
                        return password
                except (TypeError, KeyError):
                    continue
    except Exception:
        pass

    # Try wandb settings file
    try:
        from pathlib import Path
        import configparser
        wandb_settings_path = Path.home() / '.config' / 'wandb' / 'settings'
        if wandb_settings_path.exists():
            config = configparser.ConfigParser()
            config.read(wandb_settings_path)
            if 'default' in config and 'api_key' in config['default']:
                return config['default']['api_key']
    except Exception:
        pass

    return None


def _run_wandb_login(api_key):
    """Attempt to log in to wandb using the CLI or Python API."""
    import subprocess

    try:
        result = subprocess.run(
            ['wandb', 'login'],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("Wandb login successful (via 'wandb login' command)")
        else:
            print(f"First login attempt returned: {result.returncode}")
            print("Trying with --relogin flag...")
            result2 = subprocess.run(
                ['wandb', 'login', '--relogin'],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result2.returncode == 0:
                print("Wandb login successful (via 'wandb login --relogin')")
            else:
                print("Wandb login command failed")
                print(f"   Output: {result2.stdout}")
                print(f"   Error: {result2.stderr}")
                print("   Will try to proceed anyway...")
    except subprocess.TimeoutExpired:
        print("Wandb login command timed out. Will try to proceed anyway...")
    except FileNotFoundError:
        print("'wandb' command not found in PATH. Trying Python API instead...")
        try:
            wandb.login(key=api_key, relogin=True)
            print("Wandb login successful (via Python API)")
        except Exception as login_error:
            print(f"Wandb login failed: {login_error}. Will try to proceed anyway...")
    except Exception as cmd_error:
        print(f"Error running wandb login: {cmd_error}. Trying Python API instead...")
        try:
            wandb.login(key=api_key, relogin=True)
            print("Wandb login successful (via Python API)")
        except Exception as login_error:
            print(f"Wandb login failed: {login_error}. Will try to proceed anyway...")


def _generate_wandb_run_name():
    """Generate a descriptive run name based on configuration."""
    if wandb_run_name is not None:
        return wandb_run_name

    run_name_parts = []
    if use_combined_labels:
        run_name_parts.append("combined")
    else:
        run_name_parts.append("5class")
    if use_ordinal_loss:
        run_name_parts.append("ordinal")
    else:
        run_name_parts.append("mce")
    if use_features:
        run_name_parts.append("feat")
    if use_class_weights:
        run_name_parts.append("weighted")
    if train_on_all_subjects:
        run_name_parts.append("all_subjects")
    elif use_leave_one_out:
        run_name_parts.append("loso")
    elif not use_random_split:
        run_name_parts.append(f"{n_folds}fold")
    else:
        run_name_parts.append(f"{n_folds}fold_random")
    return "_".join(run_name_parts)


def initialize_wandb():
    """
    Initialize Weights & Biases logging.

    Returns True if wandb was successfully initialized, False otherwise.
    """
    if not use_wandb:
        return False

    if not WANDB_AVAILABLE:
        print("\nWandb requested but not available. Install with: pip install wandb\n")
        return False

    try:
        api_key = _resolve_wandb_api_key()

        if api_key is not None:
            os.environ['WANDB_API_KEY'] = api_key
            print(f"\n{'='*60}")
            print("Wandb API key set in environment")
            print(f"{'='*60}\n")
            _run_wandb_login(api_key)
        else:
            print(f"\n{'='*60}")
            print("No Wandb API key found")
            print("   Set wandb_api_key in configuration or run: export WANDB_API_KEY='your_key'")
            print(f"{'='*60}\n")

        if wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'
            print(f"\n{'='*60}")
            print("Wandb running in OFFLINE mode")
            print("   Run 'wandb sync' later to upload results")
            print(f"{'='*60}\n")

        run_name = _generate_wandb_run_name()

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            tags=wandb_tags,
            config={
                "use_features": use_features,
                "use_ordinal_loss": use_ordinal_loss,
                "use_class_weights": use_class_weights,
                "use_combined_labels": use_combined_labels,
                "use_random_split": use_random_split,
                "use_leave_one_out": use_leave_one_out,
                "train_on_all_subjects": train_on_all_subjects,
                "random_seed": random_seed,
                "n_folds": n_folds,
                "max_epochs": max_epochs,
                "use_early_stopping": use_early_stopping,
                "early_stopping_patience": early_stopping_patience if use_early_stopping else None,
                "min_epochs": min_epochs if use_early_stopping else None,
                "learning_rate": learning_rate,
                "batch_size": 64,
                "window_size": WINDOW_SIZE,
                "window_overlap": WINDOW_OVERLAP,
            }
        )
        print(f"\n{'='*60}")
        print("Weights & Biases initialized")
        print(f"   Project: {wandb_project}")
        print(f"   Run: {run_name}")
        print(f"{'='*60}\n")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print("Failed to initialize Weights & Biases")
        if "not logged in" in error_msg or "401" in error_msg or "Unauthorized" in error_msg:
            print("   Authentication error: You need to log in to wandb first.")
            print("   Run this command in your terminal:")
            print("      wandb login")
            print("   Or set WANDB_API_KEY environment variable")
        else:
            print(f"   Error: {error_msg}")
        print("   Continuing without wandb logging...")
        print(f"{'='*60}\n")
        return False


# ============================================================
# DATA PREPROCESSING
# ============================================================

def run_preprocessing():
    """
    Run raw data preprocessing: load raw files, bandpass filter,
    resample, window, and save to disk.
    """
    win_data_all_sub = np.empty((0, 3, WINDOW_SIZE))
    win_labels_all_sub = win_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0, WINDOW_SIZE))
    win_video_time_all_sub = np.empty((0, 1))
    NumWin = []

    for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
        try:
            data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
        except Exception:
            print(f"Can't open the file {file}")
            continue

        try:
            acc_data = data_file['acc_data'].astype('float')
        except Exception:
            try:
                acc_data = data_file['acc_data']

                def is_numeric(s):
                    try:
                        float(s)
                        return True
                    except ValueError:
                        return False

                numeric_mask = np.array([[is_numeric(cell) for cell in row] for row in acc_data])
                acc_data[~numeric_mask] = np.nan
                acc_data = acc_data.astype('float')
                acc_data = acc_data[~np.isnan(acc_data).any(axis=1)]
                if len(acc_data) == 0:
                    continue
            except Exception:
                print(f"Failed to process acc_data in {file}")
                continue

        labels = data_file.get('label_data', None)
        chorea = data_file.get('chorea_labels', None)
        video_time = data_file.get('time_data', None)
        subject_name = file.split('.')[0]

        acc_data = preprocessing.bandpass_filter(
            data=acc_data, low_cut=0.2, high_cut=15,
            sampling_rate=SRC_SAMPLE_RATE, order=4
        )

        acc_data, labels, chorea, video_time = preprocessing.resample(
            data=acc_data, labels=labels, chorea=chorea, video_time=video_time,
            original_fs=SRC_SAMPLE_RATE, target_fs=30
        )

        data, labels, chorea, video_time, shift, NumWinSub = preprocessing.data_windowing(
            data=acc_data, labels=labels, chorea=chorea, video_time=video_time,
            window_size=WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
            std_th=STD_THRESH, model_type='segmentation', subject=subject_name
        )

        win_data_all_sub = np.append(win_data_all_sub, data, axis=0)
        win_labels_all_sub = np.append(win_labels_all_sub, labels, axis=0)
        win_chorea_all_sub = np.append(win_chorea_all_sub, chorea, axis=0)
        win_shift_all_sub = np.append(win_shift_all_sub, shift, axis=0)
        win_video_time_all_sub = np.append(win_video_time_all_sub, video_time, axis=0)

        subject = np.tile(subject_name, (len(labels), 1)).reshape(-1, 1)
        win_subjects = np.append(win_subjects, subject)
        NumWin.append(NumWinSub)

        print(file, win_data_all_sub.shape)

    # Save processed data
    res = {
        'win_data_all_sub': win_data_all_sub,
        'win_labels_all_sub': win_labels_all_sub,
        'win_subjects': win_subjects,
        'win_chorea_all_sub': win_chorea_all_sub,
        'win_shift_all_sub': win_shift_all_sub,
        'win_video_time_all_sub': win_video_time_all_sub
    }

    np.savez(os.path.join(PROCESSED_DATA_DIR, 'windows_input_to_multiclass_model_new_subjects.npz'), **res)


# ============================================================
# DATA LOADING AND FILTERING
# ============================================================

def load_and_prepare_data():
    """
    Load processed data, form triple windows, filter by walking/chorea/reference/healthy,
    configure labels, and stack windows.

    Returns:
        win_acc_data: stacked triple-window accelerometer data
        win_chorea: (possibly remapped) chorea labels
        win_chorea_original: original chorea labels before remapping
        win_subjects: subject IDs for each window
        num_label_classes: number of label classes (3 or 5)
        label_list: list of label values
    """
    # --- Load processed data ---
    print("start loading input file")
    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, 'windows_input_to_multiclass_model_new_subjects.npz'))
    print("done loading input file")

    # --- Load reference file to get valid subjects ---
    reference_file = np.load(os.path.join(
        PROCESSED_DATA_DIR,
        'windows_input_to_multiclass_model_hd_only_segmentation_triple_wind_no_shift.npz'
    ))
    reference_subjects = reference_file['win_subjects']
    reference_subjects_str = np.array([str(s) for s in reference_subjects.reshape(-1)])
    valid_subjects_set = set(np.unique(reference_subjects_str))
    reference_file.close()

    print(f"\n{'='*60}")
    print("Subject Filtering (based on reference file)")
    print(f"{'='*60}")
    print("Reference file: windows_input_to_multiclass_model_hd_only_segmentation_triple_wind_no_shift.npz")
    print(f"Valid subjects in reference: {len(valid_subjects_set)}")
    print(f"  {', '.join(sorted(valid_subjects_set))}")

    # --- Extract arrays ---
    win_acc_data = input_file['win_data_all_sub']
    win_acc_data = np.transpose(win_acc_data, [0, 2, 1])
    win_labels = input_file['win_labels_all_sub']
    win_chorea = input_file['win_chorea_all_sub']
    win_subjects = input_file['win_subjects']
    win_shift = input_file['win_shift_all_sub']
    win_shift = np.mean(win_shift, axis=-1)

    # --- Form triple windows (left / mid / right) ---
    left = win_acc_data[0:-2:2]
    mid = win_acc_data[1:-1:2]
    right = win_acc_data[2::2]
    mid_labels = win_labels[1:-1:2]
    mid_chorea = win_chorea[1:-1:2]

    # --- Filter: keep only windows with walking and valid chorea ---
    walking_mask = (mid_labels > 0).sum(axis=1) > 0
    valid_chorea_mask = (mid_chorea >= 0).sum(axis=1) >= 2
    final_mask = walking_mask & valid_chorea_mask

    print(f"\n{'='*60}")
    print("Data Filtering Statistics:")
    print(f"{'='*60}")
    print(f"Total windows available:        {len(mid)}")
    print(f"Windows with walking:           {walking_mask.sum()}")
    print(f"Windows with valid chorea:      {valid_chorea_mask.sum()}")
    print(f"Windows kept (both criteria):   {final_mask.sum()}")
    print(f"Filtering ratio:                {final_mask.sum()/len(mid)*100:.1f}%")
    print(f"{'='*60}\n")

    left_wind = left[final_mask]
    mid_wind = mid[final_mask]
    right_wind = right[final_mask]
    mid_chorea = mid_chorea[final_mask]
    win_subjects = win_subjects[1:-1:2][final_mask]

    # --- Filter subjects based on reference file ---
    subjects_str = np.array([str(s) for s in win_subjects.reshape(-1)])
    subject_mask = np.array([s in valid_subjects_set for s in subjects_str])

    subjects_before = len(np.unique(subjects_str))
    subjects_after = len(np.unique(subjects_str[subject_mask]))

    print(f"\n{'='*60}")
    print("Filtering Subjects (keep only those in reference file)")
    print(f"{'='*60}")
    print(f"Subjects before filtering: {subjects_before}")
    print(f"Subjects after filtering:  {subjects_after}")

    if subjects_before > subjects_after:
        removed_subjects = set(np.unique(subjects_str)) - valid_subjects_set
        print(f"Removed subjects: {', '.join(sorted(removed_subjects))}")
        print(f"Windows before filtering: {len(left_wind)}")

    left_wind = left_wind[subject_mask]
    mid_wind = mid_wind[subject_mask]
    right_wind = right_wind[subject_mask]
    mid_chorea = mid_chorea[subject_mask]
    win_subjects = win_subjects[subject_mask]

    print(f"Windows after filtering:  {len(left_wind)}")
    print(f"Windows removed:           {len(subject_mask) - subject_mask.sum()}")
    print(f"{'='*60}\n")

    # --- Remove healthy subjects (subjects with 'CO' in their ID) ---
    subjects_str = np.array([str(s) for s in win_subjects])
    healthy_mask = np.array(['CO' in s for s in subjects_str])
    hd_mask = ~healthy_mask

    unique_before = np.unique(subjects_str)
    healthy_subjects = np.unique(subjects_str[healthy_mask])
    hd_subjects = np.unique(subjects_str[hd_mask])

    print(f"\n{'='*60}")
    print("Removing Healthy Subjects (ID contains 'CO'):")
    print(f"{'='*60}")
    print(f"Total subjects before filtering: {len(unique_before)}")
    print(f"Healthy subjects (removed): {len(healthy_subjects)}")
    if len(healthy_subjects) > 0:
        print(f"  -> {', '.join(healthy_subjects)}")
    print(f"HD subjects (kept): {len(hd_subjects)}")
    print(f"  -> {', '.join(hd_subjects)}")
    print(f"\nWindows before filtering: {len(left_wind)}")
    print(f"Windows removed (healthy): {healthy_mask.sum()}")
    print(f"Windows kept (HD): {hd_mask.sum()}")
    print(f"{'='*60}\n")

    left_wind = left_wind[hd_mask]
    mid_wind = mid_wind[hd_mask]
    right_wind = right_wind[hd_mask]
    mid_chorea = mid_chorea[hd_mask]
    win_subjects = win_subjects[hd_mask]

    win_chorea = mid_chorea
    win_chorea_original = win_chorea.copy()

    # --- Configure labels ---
    if use_combined_labels:
        print(f"\n{'='*60}")
        print("Remapping Chorea Labels to 3 Classes:")
        print(f"{'='*60}")
        print("Original label distribution:")
        for label in range(5):
            count = (win_chorea == label).sum()
            print(f"  Label {label}: {count} samples")

        win_chorea_remapped = win_chorea.copy()
        win_chorea_remapped[win_chorea == 1] = 1
        win_chorea_remapped[win_chorea == 2] = 1
        win_chorea_remapped[win_chorea == 3] = 2
        win_chorea_remapped[win_chorea == 4] = 2
        win_chorea = win_chorea_remapped

        print("\nRemapped label distribution:")
        for label in range(3):
            count = (win_chorea == label).sum()
            print(f"  Label {label}: {count} samples")
        print(f"{'='*60}\n")

        num_label_classes = 3
        label_list = [0, 1, 2]
    else:
        print(f"\n{'='*60}")
        print("Using Original 5 Chorea Labels (0-4):")
        print(f"{'='*60}")
        print("Label distribution:")
        for label in range(5):
            count = (win_chorea == label).sum()
            print(f"  Label {label}: {count} samples")
        print(f"{'='*60}\n")

        num_label_classes = 5
        label_list = [0, 1, 2, 3, 4]

    # --- Stack triple windows ---
    win_acc_data = np.stack([left_wind, mid_wind, right_wind], axis=2).reshape(-1, 3, WINDOW_SIZE * 3)

    return win_acc_data, win_chorea, win_chorea_original, win_subjects, num_label_classes, label_list


# ============================================================
# SUBJECT GROUPING AND FOLD CONFIGURATION
# ============================================================

def setup_subject_groups(win_subjects):
    """
    Convert subject arrays into group indices for cross-validation.

    Returns:
        subjects: 1D string array of subject IDs
        unique_subjects: sorted unique subject IDs
        groups: integer group indices for each sample
    """
    subjects = np.array([str(s) for s in win_subjects.reshape(-1)])
    unique_subjects = np.unique(subjects)
    unique_subjects.sort()
    subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
    groups = np.array([subject_to_idx[s] for s in subjects])
    return subjects, unique_subjects, groups


def configure_folds(subjects, unique_subjects, win_acc_data, win_chorea, groups):
    """
    Configure cross-validation folds based on the selected strategy
    (Leave-One-Out, Random GroupKFold, or Custom).

    Returns:
        fold_to_val_subjects: dict mapping fold number to list of validation subject IDs
        n_splits: number of folds
    """
    if use_leave_one_out:
        from sklearn.model_selection import LeaveOneGroupOut

        logo = LeaveOneGroupOut()
        fold_to_val_subjects = {}
        for fold_num, (train_idx, val_idx) in enumerate(logo.split(win_acc_data, win_chorea, groups), 1):
            val_subjects_in_fold = np.unique(subjects[val_idx])
            fold_to_val_subjects[fold_num] = list(val_subjects_in_fold)

        n_splits = len(fold_to_val_subjects)
        print(f"\n{'='*60}")
        print("Using LEAVE-ONE-SUBJECT-OUT (LOSO) Cross-Validation")
        print(f"Number of folds: {n_splits} (one per subject)")
        print(f"Available subjects: {', '.join(sorted(unique_subjects))}")
        print("\nFold assignments (one subject per fold):")
        for fold_num, val_subs in fold_to_val_subjects.items():
            print(f"  Fold {fold_num}: validation = {val_subs[0]}")
        print(f"{'='*60}\n")

    elif use_random_split:
        np.random.seed(random_seed)

        shuffled_subjects = unique_subjects.copy()
        np.random.shuffle(shuffled_subjects)

        subject_to_shuffled_idx = {subj: i for i, subj in enumerate(shuffled_subjects)}
        shuffled_groups = np.array([subject_to_shuffled_idx[s] for s in subjects])

        gkf = GroupKFold(n_splits=n_folds)

        fold_to_val_subjects = {}
        for fold_num, (train_idx, val_idx) in enumerate(gkf.split(win_acc_data, win_chorea, shuffled_groups), 1):
            val_subjects_in_fold = np.unique(subjects[val_idx])
            fold_to_val_subjects[fold_num] = list(val_subjects_in_fold)

        n_splits = len(fold_to_val_subjects)
        print(f"\n{'='*60}")
        print(f"Using RANDOM GroupKFold Split (seed={random_seed})")
        print(f"Number of folds: {n_splits}")
        print(f"Available subjects: {', '.join(sorted(unique_subjects))}")
        print("\nRandom fold assignments:")
        for fold_num, val_subs in fold_to_val_subjects.items():
            print(f"  Fold {fold_num}: validation = {', '.join(sorted(val_subs))}")
        print(f"{'='*60}\n")

    else:
        # Custom/Manual fold configuration
        fold_to_val_subjects = {
            1: ['IW13GHI', 'IW15GHI', 'IW6GHI', 'IW9TC'],
            2: ['IW11TC', 'IW15TC', 'IW3GHI', 'IW5TC'],
            3: ['IW10TC', 'IW12GHI', 'IW13TC', 'IW7TC', 'IW8TC'],
            4: ['IW14TC', 'IW4GHI', 'IW4TC', 'IW5GHI'],
            5: ['IW10GHI', 'IW11GHI', 'IW12TC', 'IW14GHI', 'IW6TC'],
        }

        all_specified_subjects = set()
        for fold_subjects in fold_to_val_subjects.values():
            all_specified_subjects.update(fold_subjects)

        missing_subjects = all_specified_subjects - set(unique_subjects)
        if missing_subjects:
            print(f"\nWARNING: These subjects are specified in fold_to_val_subjects but not found in data:")
            print(f"    {', '.join(sorted(missing_subjects))}")
            print(f"    Available subjects: {', '.join(sorted(unique_subjects))}")

        n_splits = len(fold_to_val_subjects)
        print(f"\n{'='*60}")
        print("Using CUSTOM Fold Configuration")
        print(f"Number of folds: {n_splits}")
        print(f"Available subjects: {', '.join(sorted(unique_subjects))}")
        print("\nFold assignments:")
        for fold_num, val_subs in fold_to_val_subjects.items():
            print(f"  Fold {fold_num}: validation = {', '.join(val_subs)}")
        print(f"{'='*60}\n")

    return fold_to_val_subjects, n_splits


# ============================================================
# FOLD DATA PREPARATION
# ============================================================

def prepare_fold_data(win_acc_data, win_chorea, window_features, subjects,
                      val_subject_list, num_label_classes, device):
    """
    Prepare train/val splits, masks, class weights, and data loaders for one fold.

    Returns a dict with:
        train_loader, val_loader, train_idx, val_idx, val_subjects,
        class_weights_tensor, X_val, y_val
    """
    val_mask = np.isin(subjects, val_subject_list)
    train_mask = ~val_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    if len(val_idx) == 0:
        print(f"  WARNING: No validation samples found for subjects {val_subject_list}. Skipping fold.")
        return None

    X_train, y_train = win_acc_data[train_idx], win_chorea[train_idx]
    X_val, y_val = win_acc_data[val_idx], win_chorea[val_idx]

    val_subjects = subjects[val_idx]
    unique_val_subjects = np.unique(val_subjects)
    print(f"\nValidation subjects ({len(unique_val_subjects)}): {', '.join(unique_val_subjects)}")

    mask_train = (y_train >= 0).astype(float)
    mask_val = (y_val >= 0).astype(float)

    y_train = np.maximum(y_train, 0)
    y_val = np.maximum(y_val, 0)

    feats_train = window_features[train_idx] if window_features is not None else None
    feats_val = window_features[val_idx] if window_features is not None else None

    # Compute class weights
    valid_train_labels = y_train[mask_train > 0]
    unique_classes, class_counts = np.unique(valid_train_labels, return_counts=True)

    print(f"\n  Class Distribution (valid samples only):")
    total_samples = len(valid_train_labels)

    if use_class_weights:
        num_total_classes = num_label_classes
        class_weights_array = np.ones(num_total_classes)
        class_weights_temp = total_samples / (len(unique_classes) * class_counts)
        for cls, weight in zip(unique_classes, class_weights_temp):
            class_weights_array[int(cls)] = weight
        class_weights_array = class_weights_array * num_total_classes / class_weights_array.sum()
        class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float32).to(device)

        for cls, count in zip(unique_classes, class_counts):
            weight = class_weights_array[int(cls)]
            print(f"    Class {int(cls)}: {count:5d} samples ({count/total_samples*100:5.1f}%) -> weight: {weight:.3f}")
    else:
        class_weights_tensor = None
        for cls, count in zip(unique_classes, class_counts):
            print(f"    Class {int(cls)}: {count:5d} samples ({count/total_samples*100:5.1f}%)")

    # Create data loaders
    train_loader = DataLoader(
        ChoreaDataset(X_train, y_train, mask_train, features=feats_train),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        ChoreaDataset(X_val, y_val, mask_val, features=feats_val),
        batch_size=64, shuffle=False
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'val_subjects': val_subjects,
        'class_weights_tensor': class_weights_tensor,
        'X_val': X_val,
        'y_val': y_val,
    }


# ============================================================
# MODEL CREATION
# ============================================================

def create_model_and_optimizer(num_classes, feat_dim, device):
    """
    Create a fresh SSL model and Adam optimizer.

    Returns (model, optimizer).
    """
    model = get_sslnet(
        tag='v1.0.0', pretrained=True,
        num_classes=num_classes, model_type='segmentation',
        padding_type='triple_wind', feat_dim=feat_dim
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def create_criterion(class_weights_tensor):
    """
    Create the loss function based on configuration.

    Returns (criterion, loss_description_string).
    """
    if use_ordinal_loss:
        criterion = CumulativeOrdinalLoss(class_weights=class_weights_tensor)
        loss_desc = "Ordinal Loss"
    else:
        criterion = MaskedCrossEntropyLoss(class_weights=class_weights_tensor)
        loss_desc = "Masked CE Loss"

    if use_class_weights:
        loss_desc += " with class weights"

    return criterion, loss_desc


# ============================================================
# TRAINING
# ============================================================

def train_model(model, optimizer, criterion, train_loader, val_loader,
                num_label_classes, device, fold_num, wandb_initialized, fold_step_offset):
    """
    Train the model for one fold with optional early stopping.

    Returns:
        model: model loaded with best weights
        best_val_loss: best validation loss achieved
        last_epoch_in_fold: last epoch number trained
        max_training_step: maximum wandb step used
    """
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    last_epoch_in_fold = 0
    max_training_step = fold_step_offset

    if use_early_stopping:
        print(f"\n  Training config: max_epochs={max_epochs}, early_stopping_patience={early_stopping_patience}, min_epochs={min_epochs}")
    else:
        print(f"\n  Training config: max_epochs={max_epochs} (early stopping disabled)")

    for epoch in range(max_epochs):
        # --- Training phase ---
        model.train()
        total_train_loss = 0
        train_outputs, train_ys, train_masks = [], [], []

        for batch in train_loader:
            if use_features:
                X_batch, y_batch, mask, feats = batch
                feats = feats.to(device)
            else:
                X_batch, y_batch, mask = batch
                feats = None

            X_batch, y_batch, mask = X_batch.to(device), y_batch.to(device).long(), mask.to(device)
            optimizer.zero_grad()
            output = model(X_batch, feats)
            loss = criterion(output, y_batch, mask)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_outputs.append(output.detach())
            train_ys.append(y_batch)
            train_masks.append(mask)

        avg_train_loss = total_train_loss / len(train_loader)
        train_auc = compute_auc_from_outputs(
            train_outputs, train_ys, train_masks,
            use_ordinal_loss, num_label_classes, device
        )
        train_auc_str = f"{train_auc:.4f}" if not np.isnan(train_auc) else "N/A"

        # --- Validation phase ---
        model.eval()
        total_val_loss = 0
        val_outputs, val_ys, val_masks = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                if use_features:
                    X_batch, y_batch, mask, feats = batch
                    feats = feats.to(device)
                else:
                    X_batch, y_batch, mask = batch
                    feats = None

                X_batch, y_batch, mask = X_batch.to(device), y_batch.to(device).long(), mask.to(device)
                output = model(X_batch, feats)
                val_loss = criterion(output, y_batch, mask)
                total_val_loss += val_loss.item()
                val_outputs.append(output)
                val_ys.append(y_batch)
                val_masks.append(mask)

        avg_val_loss = total_val_loss / len(val_loader)
        val_auc = compute_auc_from_outputs(
            val_outputs, val_ys, val_masks,
            use_ordinal_loss, num_label_classes, device
        )
        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "N/A"

        # --- Check for improvement ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improvement_marker = " (best)"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""

        print(f"Epoch {epoch + 1:3d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Train AUC: {train_auc_str} | "
              f"Val AUC: {val_auc_str} | LR: {learning_rate:.2e}{improvement_marker}")

        # --- Log to wandb ---
        if wandb_initialized:
            current_step = fold_step_offset + epoch + 1
            last_epoch_in_fold = epoch + 1
            max_training_step = max(max_training_step, current_step)
            log_dict = {
                f"fold_{fold_num}/train_loss": avg_train_loss,
                f"fold_{fold_num}/val_loss": avg_val_loss,
                f"fold_{fold_num}/learning_rate": learning_rate,
                f"fold_{fold_num}/epoch": epoch + 1,
            }
            if not np.isnan(train_auc):
                log_dict[f"fold_{fold_num}/train_auc"] = train_auc
            if not np.isnan(val_auc):
                log_dict[f"fold_{fold_num}/val_auc"] = val_auc
            if avg_val_loss < best_val_loss:
                log_dict[f"fold_{fold_num}/best_val_loss"] = best_val_loss
            wandb.log(log_dict, step=current_step)

        # --- Early stopping check ---
        if use_early_stopping and epoch >= min_epochs - 1 and epochs_without_improvement >= early_stopping_patience:
            print(f"\n  Early stopping triggered after {epoch + 1} epochs "
                  f"(no improvement for {early_stopping_patience} epochs)")
            if wandb_initialized:
                current_step = fold_step_offset + epoch + 1
                last_epoch_in_fold = epoch + 1
                max_training_step = max(max_training_step, current_step)
                wandb.log({
                    f"fold_{fold_num}/early_stopped": True,
                    f"fold_{fold_num}/epochs_trained": epoch + 1
                }, step=current_step)
            break

    # Update last_epoch_in_fold if training completed all epochs
    if epoch == max_epochs - 1:
        last_epoch_in_fold = max_epochs

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model (val_loss={best_val_loss:.4f})")

    return model, best_val_loss, last_epoch_in_fold, max_training_step


# ============================================================
# MODEL SAVING
# ============================================================

def save_fold_model(model, fold_num, val_subject_list, subjects, num_classes,
                    num_label_classes, best_val_loss, epochs_trained):
    """
    Save the trained model checkpoint for a fold.

    Returns the path where the model was saved.
    """
    model_save_dir = os.path.join(VIZUALIZE_DIR, 'saved_models')
    os.makedirs(model_save_dir, exist_ok=True)

    model_save_path = os.path.join(model_save_dir, f'model_fold{fold_num}.pth')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'fold_num': fold_num,
        'val_subjects': val_subject_list,
        'train_subjects': list(set(subjects) - set(val_subject_list)),
        'num_classes': num_classes,
        'num_label_classes': num_label_classes,
        'use_ordinal_loss': use_ordinal_loss,
        'use_combined_labels': use_combined_labels,
        'use_features': use_features,
        'use_class_weights': use_class_weights,
        'window_size': WINDOW_SIZE,
        'best_val_loss': best_val_loss,
        'epochs_trained': epochs_trained,
    }
    torch.save(checkpoint, model_save_path)
    print(f"\n  Model saved to: {model_save_path}")
    return model_save_path


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, val_loader, device):
    """
    Run inference on the validation set and collect predictions.

    Returns:
        y_pred: predicted labels (masked, flattened)
        y_true: true labels (masked, flattened)
        valid_mask: boolean mask of valid samples
        probs_valid: class probability matrix for valid samples
    """
    model.eval()
    all_preds, all_labels, all_masks, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if use_features:
                X_batch, y_batch, mask, feats = batch
                feats = feats.to(device)
            else:
                X_batch, y_batch, mask = batch
                feats = None
            X_batch = X_batch.to(device)
            logits = model(X_batch, feats).cpu()

            if use_ordinal_loss:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).sum(dim=1)
            else:
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)

            all_preds.append(pred)
            all_labels.append(y_batch)
            all_masks.append(mask)
            all_probs.append(probs)

    y_pred = torch.cat(all_preds).view(-1).numpy()
    y_true = torch.cat(all_labels).view(-1).numpy()
    all_masks_cat = torch.cat(all_masks, dim=0).view(-1)
    valid_mask = all_masks_cat.bool()

    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]

    probs_cat = torch.cat(all_probs, dim=0).permute(0, 2, 1).reshape(-1, all_probs[0].shape[1])
    probs_valid = probs_cat[valid_mask].numpy()

    return y_pred, y_true, valid_mask, probs_valid


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_fold_predictions(y_pred, y_true, probs_valid, val_subjects_per_sample, 
                         fold_num, num_label_classes, val_subject_list):
    """
    Save detailed per-sample predictions for a fold to CSV.
    
    Args:
        y_pred: predicted labels
        y_true: true labels
        probs_valid: probability matrix [n_samples, n_classes]
        val_subjects_per_sample: subject ID for each sample
        fold_num: current fold number
        num_label_classes: number of classes
        val_subject_list: list of validation subjects in this fold
    """
    # Create predictions directory
    predictions_dir = os.path.join(VIZUALIZE_DIR, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Prepare data dictionary
    data_dict = {
        'sample_id': np.arange(len(y_pred)),
        'subject_id': val_subjects_per_sample,
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': (y_pred == y_true).astype(int)
    }
    
    # Add probability columns for each class
    for class_idx in range(num_label_classes if not use_ordinal_loss else num_label_classes):
        if use_ordinal_loss:
            # For ordinal loss, probs shape is [n_samples, K-1]
            if class_idx < probs_valid.shape[1]:
                data_dict[f'prob_threshold_{class_idx+1}'] = probs_valid[:, class_idx]
        else:
            # For softmax, probs shape is [n_samples, K]
            if class_idx < probs_valid.shape[1]:
                data_dict[f'prob_class_{class_idx}'] = probs_valid[:, class_idx]
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Sort by subject_id and sample_id for better readability
    df = df.sort_values(['subject_id', 'sample_id']).reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(predictions_dir, f'predictions_fold{fold_num}.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    print(f"\n  Saved predictions to: {csv_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Validation subjects: {', '.join(sorted(val_subject_list))}")
    
    # Also save a summary per subject
    summary_data = []
    for subject in sorted(df['subject_id'].unique()):
        subject_df = df[df['subject_id'] == subject]
        summary_data.append({
            'subject_id': subject,
            'total_samples': len(subject_df),
            'correct_predictions': subject_df['correct'].sum(),
            'accuracy': subject_df['correct'].mean(),
            'true_label_dist': subject_df['true_label'].value_counts().to_dict(),
            'pred_label_dist': subject_df['predicted_label'].value_counts().to_dict()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(predictions_dir, f'predictions_summary_fold{fold_num}.csv')
    
    # Flatten the distribution dicts for CSV
    summary_export = summary_df[['subject_id', 'total_samples', 'correct_predictions', 'accuracy']].copy()
    summary_export.to_csv(summary_path, index=False, float_format='%.6f')
    
    print(f"  Saved summary to: {summary_path}")
    
    return csv_path, summary_path


# ============================================================
# VISUALIZATION: CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_true, y_pred, label_list, fold_num,
                          acc, prec, rec, f1, auc_str, loss_name,
                          wandb_initialized=False, eval_step=None):
    """
    Plot and save a confusion matrix with margins for a single fold.

    Returns the matplotlib figure.
    """
    labels = label_list
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    total_sum = np.sum(cm)

    display_labels = labels + ['Total']
    n_classes = len(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    ax.set_xlim(-0.5, n_classes + 0.5)
    ax.set_ylim(n_classes + 0.5, -0.5)

    for i in range(n_classes):
        rect = plt.Rectangle((n_classes - 0.5, i - 0.5), 1, 1,
                              facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(n_classes, i, format(row_sums[i], 'd'),
                ha="center", va="center", color="black", fontsize=12)

        rect = plt.Rectangle((i - 0.5, n_classes - 0.5), 1, 1,
                              facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(i, n_classes, format(col_sums[i], 'd'),
                ha="center", va="center", color="black", fontsize=12)

    rect = plt.Rectangle((n_classes - 0.5, n_classes - 0.5), 1, 1,
                          facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(n_classes, n_classes, format(total_sum, 'd'),
            ha="center", va="center", color="black", fontsize=12)

    ax.set_xticks(np.arange(n_classes + 1))
    ax.set_yticks(np.arange(n_classes + 1))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)

    title_parts = [loss_name]
    if use_class_weights:
        title_parts.append("Weighted")
    loss_title = " + ".join(title_parts) if len(title_parts) > 1 else title_parts[0]

    metrics_str = (
        f"Accuracy: {acc:.3f} | "
        f"Precision: {prec:.3f} | "
        f"Recall: {rec:.3f} | "
        f"F1: {f1:.3f} | "
        f"AUC: {auc_str}"
    )
    ax.set_title(f"Confusion Matrix - {loss_title}\n{metrics_str}", fontsize=12)

    plt.tight_layout()
    conf_matrix_path = VIZUALIZE_DIR + f"conf_matrix_fold{fold_num}.png"
    plt.savefig(conf_matrix_path)

    if wandb_initialized and eval_step is not None:
        wandb.log({f"fold_{fold_num}/confusion_matrix": wandb.Image(fig)}, step=eval_step)

    plt.show()
    return fig


# ============================================================
# VISUALIZATION: ERROR ANALYSIS
# ============================================================

def run_error_analysis(y_pred, y_true, y_true_original, val_subjects_per_sample, fold_num, label_list):
    """
    Analyze prediction errors for a fold.
    For combined labels: analyze 1->0 errors by original label.
    """
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS - Fold {fold_num}")
    print(f"{'='*60}")

    # 1->0 error analysis (only meaningful for combined labels)
    if use_combined_labels:
        errors_1to0_mask = (y_pred == 0) & (y_true == 1)

        if errors_1to0_mask.sum() > 0:
            original_labels_of_errors = y_true_original[errors_1to0_mask]

            orig_1_errors = (original_labels_of_errors == 1).sum()
            orig_2_errors = (original_labels_of_errors == 2).sum()
            total_errors = errors_1to0_mask.sum()

            total_orig_1 = (y_true_original == 1).sum()
            total_orig_2 = (y_true_original == 2).sum()

            error_rate_1 = (orig_1_errors / total_orig_1 * 100) if total_orig_1 > 0 else 0
            error_rate_2 = (orig_2_errors / total_orig_2 * 100) if total_orig_2 > 0 else 0

            print(f"\n1->0 Error Analysis:")
            print(f"  Total errors (predicted 0, true 1): {total_errors}")
            print(f"  Originally label 1: {orig_1_errors} ({orig_1_errors/total_errors*100:.1f}% of errors)")
            print(f"    -> Error rate: {orig_1_errors}/{total_orig_1} = {error_rate_1:.1f}%")
            print(f"  Originally label 2: {orig_2_errors} ({orig_2_errors/total_errors*100:.1f}% of errors)")
            print(f"    -> Error rate: {orig_2_errors}/{total_orig_2} = {error_rate_2:.1f}%")

            # Visualize
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            categories = ['Original\nLabel 1', 'Original\nLabel 2']
            counts = [orig_1_errors, orig_2_errors]
            colors = ['#FF6B6B', '#4ECDC4']

            bars1 = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            for bar, count in zip(bars1, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                         f'{count}\n({count/total_errors*100:.1f}%)',
                         ha='center', va='bottom', fontsize=11)

            ax1.set_ylabel('Number of Errors', fontsize=12)
            ax1.set_title(f'Error Counts\n(Total: {total_errors} errors)', fontsize=12)
            ax1.set_ylim(0, max(counts) * 1.25)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')

            error_rates = [error_rate_1, error_rate_2]
            bars2 = ax2.bar(categories, error_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            for bar, err_count, total_count, rate in zip(bars2, counts, [total_orig_1, total_orig_2], error_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{rate:.1f}%\n({err_count}/{total_count})',
                         ha='center', va='bottom', fontsize=11)

            ax2.set_ylabel('Error Rate (%)', fontsize=12)
            ax2.set_title('Error Rates\n(Errors / Total samples per label)', fontsize=12)
            ax2.set_ylim(0, max(error_rates) * 1.25)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
            ax2.legend()

            fig.suptitle(f'Fold {fold_num}: 1->0 Error Analysis by Original Label', fontsize=14)

            plt.tight_layout()
            error_analysis_path = VIZUALIZE_DIR + f"error_analysis_1to0_fold{fold_num}.png"
            plt.savefig(error_analysis_path)
            plt.show()
            print(f"  Saved error analysis to: {error_analysis_path}")


# ============================================================
# VISUALIZATION: PER-SUBJECT DISTRIBUTION
# ============================================================

def plot_per_subject_distribution(y_pred, y_true, val_subjects_per_sample, fold_num, label_list):
    """
    Plot per-subject histograms comparing true vs predicted label distributions.
    """
    print(f"\n2. Per-Subject Distribution Analysis:")

    unique_val_subs = np.unique(val_subjects_per_sample)
    n_subjects = len(unique_val_subs)

    n_cols = min(3, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_subjects == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, subject in enumerate(unique_val_subs):
        ax = axes[idx]

        subject_mask = val_subjects_per_sample == subject
        subject_preds = y_pred[subject_mask]
        subject_true = y_true[subject_mask]

        labels_range = label_list
        pred_counts = [np.sum(subject_preds == i) for i in labels_range]
        true_counts = [np.sum(subject_true == i) for i in labels_range]

        x = np.arange(len(labels_range))
        width = 0.35

        bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels',
                        color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predictions',
                        color='#e74c3c', alpha=0.8, edgecolor='black')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Chorea Level', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Subject: {subject}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_range)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        subject_acc = accuracy_score(subject_true, subject_preds)
        ax.text(0.02, 0.98, f'Acc: {subject_acc:.2f}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        print(f"  Subject {subject}: {len(subject_true)} samples, Accuracy: {subject_acc:.3f}")

    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Fold {fold_num}: Per-Subject Label Distribution (True vs Predicted)',
                 fontsize=14, y=1.00)
    plt.tight_layout()

    subject_hist_path = VIZUALIZE_DIR + f"per_subject_distribution_fold{fold_num}.png"
    plt.savefig(subject_hist_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved per-subject histograms to: {subject_hist_path}")

    print(f"{'='*60}\n")


# ============================================================
# AGGREGATED RESULTS
# ============================================================

def compute_and_display_aggregated_results(all_folds_y_pred, all_folds_y_true, all_folds_probs,
                                           fold_results, fold_to_val_subjects,
                                           n_splits, label_list, num_label_classes,
                                           loss_name, wandb_initialized, max_training_step,
                                           all_folds_subjects=None):
    """
    Compute and display aggregated metrics from all folds.
    Plot the aggregated confusion matrix and print the classification report.
    
    Args:
        all_folds_subjects: optional list of subject arrays for each fold (for saving predictions)
    """
    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS FROM ALL {n_splits} FOLDS")
    print(f"{'='*60}")

    all_y_pred = np.concatenate(all_folds_y_pred)
    all_y_true = np.concatenate(all_folds_y_true)
    pooled_probs = np.concatenate(all_folds_probs, axis=0)

    pooled_auc = auc_from_probs(pooled_probs, all_y_true, use_ordinal_loss, num_label_classes)
    pooled_auc_str = f"{pooled_auc:.3f}" if not np.isnan(pooled_auc) else "N/A"

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)

    fold_aucs = [auc for (_, _, _, _, _, auc) in fold_results if not np.isnan(auc)]
    mean_auc_val = np.mean(fold_aucs) if fold_aucs else np.nan
    std_auc_val = np.std(fold_aucs) if len(fold_aucs) > 1 else 0.0
    agg_auc_str = f"{mean_auc_val:.3f} +/- {std_auc_val:.3f}" if not np.isnan(mean_auc_val) else "N/A"

    print(f"\nOverall Metrics (aggregated from all folds):")
    print(f"  Total samples: {len(all_y_true)}")
    print(f"  Accuracy:  {overall_acc:.3f}")
    print(f"  Precision: {overall_prec:.3f}")
    print(f"  Recall:    {overall_rec:.3f}")
    print(f"  F1-Score:  {overall_f1:.3f}")
    print(f"  AUC (pooled): {pooled_auc_str}  |  AUC (mean+/-std over folds): {agg_auc_str}")

    # Save aggregated predictions if subject info is provided
    if all_folds_subjects is not None:
        _save_aggregated_predictions(
            all_y_pred, all_y_true, pooled_probs, 
            np.concatenate(all_folds_subjects),
            num_label_classes, n_splits
        )

    # Per-fold summary
    print(f"\nPer-Fold Results:")
    print(f"{'-'*72}")
    print(f"{'Fold':<10} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'AUC':>12}")
    print(f"{'-'*72}")

    fold_metrics_array = np.array([
        (acc, prec, rec, f1, auc if not np.isnan(auc) else np.nan)
        for (fold_num, acc, prec, rec, f1, auc) in fold_results
    ])

    for (fold_num, acc, prec, rec, f1, auc) in fold_results:
        auc_disp = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        print(f"{'Fold ' + str(fold_num):<10} {acc:>12.3f} {prec:>12.3f} {rec:>12.3f} {f1:>12.3f} {auc_disp:>12}")

    mean_acc = np.nanmean(fold_metrics_array[:, 0])
    std_acc = np.nanstd(fold_metrics_array[:, 0])
    mean_prec = np.nanmean(fold_metrics_array[:, 1])
    std_prec = np.nanstd(fold_metrics_array[:, 1])
    mean_rec = np.nanmean(fold_metrics_array[:, 2])
    std_rec = np.nanstd(fold_metrics_array[:, 2])
    mean_f1 = np.nanmean(fold_metrics_array[:, 3])
    std_f1 = np.nanstd(fold_metrics_array[:, 3])
    mean_auc = np.nanmean(fold_metrics_array[:, 4])
    std_auc = np.nanstd(fold_metrics_array[:, 4])

    # Log per-fold summary table to wandb
    if wandb_initialized:
        fold_table_data = []
        for (fold_num, acc, prec, rec, f1, auc) in fold_results:
            fold_table_data.append([fold_num, acc, prec, rec, f1, auc if not np.isnan(auc) else None])
        fold_table = wandb.Table(
            columns=["Fold", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
            data=fold_table_data
        )
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log({"overall/per_fold_summary": fold_table}, step=summary_step)

    # Identify best fold
    best_fold_idx = np.argmax(fold_metrics_array[:, 3])
    best_fold_num = fold_results[best_fold_idx][0]
    best_f1 = fold_metrics_array[best_fold_idx, 3]

    # Log aggregated metrics to wandb
    if wandb_initialized:
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        log_overall = {
            "overall/accuracy": overall_acc,
            "overall/precision": overall_prec,
            "overall/recall": overall_rec,
            "overall/f1_score": overall_f1,
            "overall/mean_accuracy": mean_acc,
            "overall/std_accuracy": std_acc,
            "overall/mean_precision": mean_prec,
            "overall/std_precision": std_prec,
            "overall/mean_recall": mean_rec,
            "overall/std_recall": std_rec,
            "overall/mean_f1": mean_f1,
            "overall/std_f1": std_f1,
            "overall/best_fold": best_fold_num,
            "overall/best_f1": best_f1,
        }
        if not np.isnan(mean_auc):
            log_overall["overall/mean_auc"] = mean_auc
            log_overall["overall/std_auc"] = std_auc
        if not np.isnan(pooled_auc):
            log_overall["overall/pooled_auc"] = pooled_auc
        wandb.log(log_overall, step=summary_step)

    print(f"{'-'*72}")
    mean_auc_disp = f"{mean_auc:.3f}" if not np.isnan(mean_auc) else "N/A"
    std_auc_disp = f"{std_auc:.3f}" if not np.isnan(mean_auc) else "N/A"
    print(f"{'Mean':<10} {mean_acc:>12.3f} {mean_prec:>12.3f} {mean_rec:>12.3f} {mean_f1:>12.3f} {mean_auc_disp:>12}")
    print(f"{'Std':<10} {std_acc:>12.3f} {std_prec:>12.3f} {std_rec:>12.3f} {std_f1:>12.3f} {std_auc_disp:>12}")
    print(f"{'='*72}")

    # --- Aggregated confusion matrix ---
    _plot_aggregated_confusion_matrix(
        all_y_true, all_y_pred, label_list, n_splits, loss_name,
        overall_acc, overall_prec, overall_rec, overall_f1, pooled_auc_str,
        wandb_initialized, max_training_step
    )

    # Print classification report
    print(f"\nClassification Report (Aggregated):")
    print(classification_report(all_y_true, all_y_pred, labels=label_list, zero_division=0))

    # Print saved models summary
    _print_saved_models_summary(fold_to_val_subjects, best_fold_num, best_f1)

    # Finish wandb
    if wandb_initialized:
        wandb.finish()
        print("Wandb run completed and synced")


def _save_aggregated_predictions(all_y_pred, all_y_true, pooled_probs, 
                                 all_subjects, num_label_classes, n_splits):
    """
    Save aggregated predictions from all folds to CSV.
    
    Args:
        all_y_pred: concatenated predictions from all folds
        all_y_true: concatenated true labels from all folds
        pooled_probs: concatenated probabilities from all folds
        all_subjects: concatenated subject IDs from all folds
        num_label_classes: number of classes
        n_splits: number of folds
    """
    predictions_dir = os.path.join(VIZUALIZE_DIR, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Prepare data dictionary
    data_dict = {
        'sample_id': np.arange(len(all_y_pred)),
        'subject_id': all_subjects,
        'true_label': all_y_true,
        'predicted_label': all_y_pred,
        'correct': (all_y_pred == all_y_true).astype(int)
    }
    
    # Add probability columns for each class
    for class_idx in range(num_label_classes if not use_ordinal_loss else num_label_classes):
        if use_ordinal_loss:
            if class_idx < pooled_probs.shape[1]:
                data_dict[f'prob_threshold_{class_idx+1}'] = pooled_probs[:, class_idx]
        else:
            if class_idx < pooled_probs.shape[1]:
                data_dict[f'prob_class_{class_idx}'] = pooled_probs[:, class_idx]
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    df = df.sort_values(['subject_id', 'sample_id']).reset_index(drop=True)
    
    # Save aggregated predictions
    cv_type = "LOSO" if use_leave_one_out else f"{n_splits}fold"
    csv_path = os.path.join(predictions_dir, f'predictions_aggregated_all_folds_{cv_type}.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    print(f"\n{'='*60}")
    print(f"SAVED AGGREGATED PREDICTIONS")
    print(f"{'='*60}")
    print(f"  File: {csv_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Subjects: {len(df['subject_id'].unique())}")
    print(f"  Overall accuracy: {df['correct'].mean():.4f}")
    
    # Save per-subject summary
    summary_data = []
    for subject in sorted(df['subject_id'].unique()):
        subject_df = df[df['subject_id'] == subject]
        summary_data.append({
            'subject_id': subject,
            'total_samples': len(subject_df),
            'correct_predictions': subject_df['correct'].sum(),
            'accuracy': subject_df['correct'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(predictions_dir, f'predictions_summary_all_folds_{cv_type}.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}\n")


def _plot_aggregated_confusion_matrix(all_y_true, all_y_pred, label_list, n_splits, loss_name,
                                      overall_acc, overall_prec, overall_rec, overall_f1,
                                      pooled_auc_str, wandb_initialized, max_training_step):
    """Plot the aggregated confusion matrix from all folds."""
    labels = label_list
    cm_agg = confusion_matrix(all_y_true, all_y_pred, labels=labels)

    row_sums = np.sum(cm_agg, axis=1)
    col_sums = np.sum(cm_agg, axis=0)
    total_sum = np.sum(cm_agg)

    display_labels = labels + ['Total']
    n_classes = len(labels)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_agg, interpolation='nearest', cmap='Blues')

    thresh = cm_agg.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm_agg[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_agg[i, j] > thresh else "black",
                    fontsize=12)

    ax.set_xlim(-0.5, n_classes + 0.5)
    ax.set_ylim(n_classes + 0.5, -0.5)

    for i in range(n_classes):
        rect = plt.Rectangle((n_classes - 0.5, i - 0.5), 1, 1,
                              facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(n_classes, i, format(row_sums[i], 'd'),
                ha="center", va="center", color="black", fontsize=12)

        rect = plt.Rectangle((i - 0.5, n_classes - 0.5), 1, 1,
                              facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(i, n_classes, format(col_sums[i], 'd'),
                ha="center", va="center", color="black", fontsize=12)

    rect = plt.Rectangle((n_classes - 0.5, n_classes - 0.5), 1, 1,
                          facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(n_classes, n_classes, format(total_sum, 'd'),
            ha="center", va="center", color="black", fontsize=12)

    ax.set_xticks(np.arange(n_classes + 1))
    ax.set_yticks(np.arange(n_classes + 1))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)

    # Per-class metrics
    per_class_precision = precision_score(all_y_true, all_y_pred, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(all_y_true, all_y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(all_y_true, all_y_pred, average=None, labels=labels, zero_division=0)

    if wandb_initialized:
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[f"overall/class_{label}_precision"] = per_class_precision[i]
            per_class_metrics[f"overall/class_{label}_recall"] = per_class_recall[i]
            per_class_metrics[f"overall/class_{label}_f1"] = per_class_f1[i]
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log(per_class_metrics, step=summary_step)

    title_parts = [loss_name]
    if use_class_weights:
        title_parts.append("Weighted")
    loss_title = " + ".join(title_parts) if len(title_parts) > 1 else title_parts[0]

    metrics_str = (
        f"Accuracy: {overall_acc:.3f} | "
        f"Precision: {overall_prec:.3f} | "
        f"Recall: {overall_rec:.3f} | "
        f"F1: {overall_f1:.3f} | "
        f"AUC (pooled): {pooled_auc_str}"
    )

    ax.set_title(f"AGGREGATED Confusion Matrix ({n_splits}-Fold CV) - {loss_title}\n"
                 f"Total Samples: {len(all_y_true)}\n{metrics_str}", fontsize=12)

    class_metrics_text = "Per-Class Metrics:\n"
    for i, label in enumerate(labels):
        class_metrics_text += f"  Class {label}: Prec={per_class_precision[i]:.3f}, Rec={per_class_recall[i]:.3f}, F1={per_class_f1[i]:.3f}\n"

    plt.figtext(0.5, 0.02, class_metrics_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    agg_cm_path = VIZUALIZE_DIR + "aggregated_confusion_matrix.png"
    plt.savefig(agg_cm_path, bbox_inches='tight', dpi=150)

    if wandb_initialized:
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log({"overall/aggregated_confusion_matrix": wandb.Image(fig)}, step=summary_step)

    plt.show()
    print(f"\nSaved aggregated confusion matrix to: {agg_cm_path}")


def _print_saved_models_summary(fold_to_val_subjects, best_fold_num, best_f1):
    """Print a summary of all saved models and usage instructions."""
    model_save_dir = os.path.join(VIZUALIZE_DIR, 'saved_models')
    print(f"\n{'='*60}")
    print("SAVED MODELS SUMMARY")
    print(f"{'='*60}")
    print(f"Models saved to: {model_save_dir}")
    print(f"\nAvailable models:")
    for fold_num, val_subs in fold_to_val_subjects.items():
        model_path = os.path.join(model_save_dir, f'model_fold{fold_num}.pth')
        if os.path.exists(model_path):
            print(f"  - model_fold{fold_num}.pth (validation subjects: {', '.join(val_subs)})")

    print(f"\nBest model: Fold {best_fold_num} (F1 = {best_f1:.3f})")
    best_model_path = os.path.join(model_save_dir, f'model_fold{best_fold_num}.pth')
    print(f"   Path: {best_model_path}")

    print(f"\n{'-'*60}")
    print("TO LOAD A MODEL FOR VALIDATION ON NEW SUBJECTS:")
    print(f"{'-'*60}")
    print(f"""
# Example code to load and use a saved model:

import torch
from sslmodel import get_sslnet

# Load the checkpoint
checkpoint = torch.load('{best_model_path}')

# Recreate the model with same configuration
model = get_sslnet(
    tag='v1.0.0', 
    pretrained=False,
    num_classes=checkpoint['num_classes'], 
    model_type='segmentation',
    padding_type='triple_wind', 
    feat_dim=0  # Set to feature dimension if use_features=True
)

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Now you can run inference on new data
# output = model(X_new, features=None)
""")
    print(f"{'='*60}")
    print("Cross-validation complete!")
    print(f"{'='*60}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Initialize wandb ---
    wandb_initialized = initialize_wandb()

    # --- Preprocessing (if enabled) ---
    if preprocessing_mode:
        run_preprocessing()

    # --- Load and prepare data ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    win_acc_data, win_chorea, win_chorea_original, win_subjects, num_label_classes, label_list = \
        load_and_prepare_data()

    # --- Extract features ---
    window_features = compute_all_window_features(win_acc_data)

    # --- Setup subject groups ---
    subjects, unique_subjects, groups = setup_subject_groups(win_subjects)

    # --- Configure folds ---
    fold_to_val_subjects, n_splits = configure_folds(
        subjects, unique_subjects, win_acc_data, win_chorea, groups
    )

    # --- Train on all subjects (no CV) ---
    if train_on_all_subjects:
        print(f"\n{'='*60}")
        print("TRAINING ON ALL SUBJECTS MODE")
        print(f"{'='*60}")
        print("Skipping cross-validation. Training on all available subjects.")
        print(f"{'='*60}\n")

        if use_ordinal_loss:
            num_classes = num_label_classes - 1
        else:
            num_classes = num_label_classes

        model, checkpoint = train_on_all_subjects_function(
            win_acc_data, win_chorea, win_subjects, window_features,
            num_label_classes, label_list, device, wandb_initialized
        )

        if wandb_initialized:
            wandb.finish()
            print("Wandb run completed and synced")
        return

    # --- Determine loss configuration ---
    if use_ordinal_loss:
        num_classes = num_label_classes - 1
        loss_name = "Ordinal"
        print(f"\n{'='*60}")
        print(f"Using Cumulative Ordinal Loss with {num_classes} outputs for {num_label_classes} classes")
        print(f"{'='*60}")
    else:
        num_classes = num_label_classes
        loss_name = "MaskedCE"
        print(f"\n{'='*60}")
        print(f"Using Masked Cross-Entropy Loss with {num_classes} outputs for {num_label_classes} classes")
        print(f"{'='*60}")

    # --- Cross-validation loop ---
    fold_results = []
    all_folds_y_pred = []
    all_folds_y_true = []
    all_folds_probs = []
    all_folds_subjects = []
    max_training_step = 0

    for fold_num, val_subject_list in fold_to_val_subjects.items():
        print(f"\n=== Fold {fold_num}/{n_splits} ===")

        fold_step_offset = max_training_step

        # Prepare fold data
        fold_data = prepare_fold_data(
            win_acc_data, win_chorea, window_features, subjects,
            val_subject_list, num_label_classes, device
        )
        if fold_data is None:
            continue

        train_loader = fold_data['train_loader']
        val_loader = fold_data['val_loader']
        val_idx = fold_data['val_idx']
        val_subjects = fold_data['val_subjects']
        class_weights_tensor = fold_data['class_weights_tensor']
        y_val = fold_data['y_val']

        if class_weights_tensor is not None:
            print(f"  Class weights tensor: shape={class_weights_tensor.shape}, values={class_weights_tensor}")

        # Create model and criterion
        feat_dim = window_features.shape[1] if use_features else 0
        model, optimizer = create_model_and_optimizer(num_classes, feat_dim, device)
        criterion, loss_desc = create_criterion(class_weights_tensor)
        print(f"  Using {loss_desc}")

        # Train
        model, best_val_loss, last_epoch_in_fold, max_training_step = train_model(
            model, optimizer, criterion, train_loader, val_loader,
            num_label_classes, device, fold_num, wandb_initialized, fold_step_offset
        )

        # Save model
        save_fold_model(
            model, fold_num, val_subject_list, subjects,
            num_classes, num_label_classes, best_val_loss, last_epoch_in_fold
        )

        # Evaluate
        y_pred, y_true, valid_mask, probs_valid = evaluate_model(model, val_loader, device)

        # Collect results for aggregation
        all_folds_y_pred.append(y_pred)
        all_folds_y_true.append(y_true)
        all_folds_probs.append(probs_valid)

        # Get original labels and subjects for error analysis
        y_true_original = win_chorea_original[val_idx].reshape(-1)[valid_mask.numpy()]
        val_subjects_per_sample = np.repeat(val_subjects, y_val.shape[1])[valid_mask.numpy()]
        
        # Store subjects for aggregated predictions
        all_folds_subjects.append(val_subjects_per_sample)

        # Save predictions to CSV
        save_fold_predictions(
            y_pred, y_true, probs_valid, val_subjects_per_sample,
            fold_num, num_label_classes, val_subject_list
        )

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        auc = auc_from_probs(probs_valid, y_true, use_ordinal_loss, num_label_classes)
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"

        print(f"Fold {fold_num} -> Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc_str}")
        fold_results.append((fold_num, acc, prec, rec, f1, auc))

        # Log fold metrics to wandb
        if wandb_initialized:
            eval_step = fold_step_offset + last_epoch_in_fold + 1
            max_training_step = max(max_training_step, eval_step)
            log_dict = {
                f"fold_{fold_num}/accuracy": acc,
                f"fold_{fold_num}/precision": prec,
                f"fold_{fold_num}/recall": rec,
                f"fold_{fold_num}/f1_score": f1,
            }
            if not np.isnan(auc):
                log_dict[f"fold_{fold_num}/auc"] = auc
            wandb.log(log_dict, step=eval_step)

        # Plot confusion matrix
        cm_eval_step = fold_step_offset + last_epoch_in_fold + 2
        max_training_step = max(max_training_step, cm_eval_step) if wandb_initialized else max_training_step
        plot_confusion_matrix(
            y_true, y_pred, label_list, fold_num,
            acc, prec, rec, f1, auc_str, loss_name,
            wandb_initialized=wandb_initialized, eval_step=cm_eval_step
        )

        # Error analysis
        run_error_analysis(y_pred, y_true, y_true_original, val_subjects_per_sample, fold_num, label_list)

        # Per-subject distribution
        plot_per_subject_distribution(y_pred, y_true, val_subjects_per_sample, fold_num, label_list)

    # --- Aggregated results ---
    compute_and_display_aggregated_results(
        all_folds_y_pred, all_folds_y_true, all_folds_probs,
        fold_results, fold_to_val_subjects,
        n_splits, label_list, num_label_classes,
        loss_name, wandb_initialized, max_training_step,
        all_folds_subjects=all_folds_subjects
    )


if __name__ == '__main__':
    if validation_only_mode:
        validate_model(validation_model_path, validation_subjects, WINDOW_SIZE, PROCESSED_DATA_DIR, VIZUALIZE_DIR)
    else:
        main()
