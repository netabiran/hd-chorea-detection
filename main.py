import preprocessing
import numpy as np
import os
import torch
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sslmodel import get_sslnet
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupKFold

RAW_DATA_AND_LABELS_DIR = '/home/netabiran/data_ready/hd_dataset/lab_geneactive/synced_labeled_data'

preprocessing_mode = False
use_features = False
use_ordinal_loss = True  # True: ordinal loss (K-1 outputs for K classes), False: masked cross-entropy (K outputs)
use_ssl_encoder = False  # True: load SSL-trained encoder if available, False: use original pretrained
use_class_weights = False  # True: use class weights to handle imbalanced data, False: no weighting
use_focal_loss = False  # True: use focal loss (only for masked CE), False: standard loss
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
use_wandb = True  # True: enable wandb logging, False: disable
wandb_project = "hd-chorea-detection"  # Wandb project name
wandb_entity = None  # Wandb entity/team (None for personal account)
wandb_run_name = "new_labels_combined_logo_ordinal_no_features_no_early_stopping"  # Custom run name (None for auto-generated)
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

VIZUALIZE_DIR = "/home/netabiran/hd-chorea-detection/figures_output/final_results/new_labels_combined_logo_ordinal_no_features_no_early_stopping/"
os.makedirs(VIZUALIZE_DIR, exist_ok=True)

SRC_SAMPLE_RATE = int(100)
STD_THRESH = 0.1
WINDOW_SIZE = int(30*10)
WINDOW_OVERLAP = int(30*5)  


def validate_model():
    """
    Validation-only mode: Load a saved model and evaluate on specific subjects.
    Uses the same preprocessing pipeline as training.
    """
    from torch.utils.data import DataLoader, Dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"VALIDATION-ONLY MODE")
    print(f"{'='*60}")
    print(f"Model path: {validation_model_path}")
    print(f"Validation subjects: {', '.join(validation_subjects)}")
    print(f"{'='*60}\n")
    
    # Check if model exists
    if not os.path.exists(validation_model_path):
        print(f"❌ ERROR: Model file not found at {validation_model_path}")
        return
    
    # Load checkpoint and get configuration
    checkpoint = torch.load(validation_model_path, map_location=device)
    
    print(f"Model Configuration from checkpoint:")
    print(f"  • num_classes: {checkpoint.get('num_classes', 'N/A')}")
    print(f"  • num_label_classes: {checkpoint.get('num_label_classes', 'N/A')}")
    print(f"  • use_ordinal_loss: {checkpoint.get('use_ordinal_loss', 'N/A')}")
    print(f"  • use_combined_labels: {checkpoint.get('use_combined_labels', 'N/A')}")
    print(f"  • use_features: {checkpoint.get('use_features', 'N/A')}")
    print(f"  • Original train subjects: {len(checkpoint.get('train_subjects', []))} subjects")
    print(f"  • Original val subjects: {checkpoint.get('val_subjects', [])}")
    
    # Get configuration from checkpoint
    num_classes = checkpoint.get('num_classes', 2)
    num_label_classes = checkpoint.get('num_label_classes', 3)
    model_use_ordinal_loss = checkpoint.get('use_ordinal_loss', True)
    model_use_combined_labels = checkpoint.get('use_combined_labels', True)
    model_use_features = checkpoint.get('use_features', False)
    
    # Determine label list based on configuration
    if model_use_combined_labels:
        label_list = [0, 1, 2]
    else:
        label_list = [0, 1, 2, 3, 4]
    
    # ============================================================
    # LOAD AND PREPROCESS DATA
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Loading and preprocessing data...")
    print(f"{'='*60}")
    
    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_new_subjects.npz'))
    
    win_acc_data = input_file['win_data_all_sub']
    win_acc_data = np.transpose(win_acc_data, [0, 2, 1])
    win_labels = input_file['win_labels_all_sub']
    win_chorea = input_file['win_chorea_all_sub']
    win_subjects = input_file['win_subjects']
    
    # Triple window preparation
    left = win_acc_data[0:-2:2]
    mid = win_acc_data[1:-1:2]
    right = win_acc_data[2::2]
    mid_labels = win_labels[1:-1:2]
    mid_chorea = win_chorea[1:-1:2]
    
    # Apply same filtering as training
    walking_mask = (mid_labels > 0).sum(axis=1) > 0
    valid_chorea_mask = (mid_chorea >= 0).sum(axis=1) >= 2
    final_mask = walking_mask & valid_chorea_mask
    
    left_wind = left[final_mask]
    mid_wind = mid[final_mask]
    right_wind = right[final_mask]
    mid_chorea = mid_chorea[final_mask]
    win_subjects = win_subjects[1:-1:2][final_mask]
    
    # Remove healthy subjects
    subjects_str = np.array([str(s) for s in win_subjects])
    healthy_mask = np.array(['CO' in s for s in subjects_str])
    hd_mask = ~healthy_mask
    
    left_wind = left_wind[hd_mask]
    mid_wind = mid_wind[hd_mask]
    right_wind = right_wind[hd_mask]
    mid_chorea = mid_chorea[hd_mask]
    win_subjects = win_subjects[hd_mask]
    
    win_chorea = mid_chorea
    win_chorea_original = win_chorea.copy()
    
    # Apply label remapping if needed
    if model_use_combined_labels:
        win_chorea_remapped = win_chorea.copy()
        win_chorea_remapped[win_chorea == 1] = 1
        win_chorea_remapped[win_chorea == 2] = 1
        win_chorea_remapped[win_chorea == 3] = 2
        win_chorea_remapped[win_chorea == 4] = 2
        win_chorea = win_chorea_remapped
    
    # Stack windows
    win_acc_data = np.stack([left_wind, mid_wind, right_wind], axis=2).reshape(-1, 3, WINDOW_SIZE * 3)
    
    # Extract features if needed
    if model_use_features:
        from scipy.stats import skew, kurtosis, entropy
        from scipy.fft import rfft, rfftfreq
        
        def extract_features(acc_window, fs=30):
            features = []
            for axis in range(acc_window.shape[0]):
                x = acc_window[axis]
                features.append(x.mean())
                features.append(x.std())
                v = np.diff(x, n=1, prepend=x[0])
                j = np.diff(v, n=1, prepend=v[0])
                features.append(v.mean())
                features.append(v.std())
                features.append(j.mean())
                features.append(j.std())
            for axis in range(acc_window.shape[0]):
                x = acc_window[axis]
                X = np.fft.rfft(x)
                freqs = np.fft.rfftfreq(len(x), 1/fs)
                psd = np.abs(X)**2
                bands = [(0.5, 3), (3, 7), (7, 12)]
                for low, high in bands:
                    mask = (freqs >= low) & (freqs < high)
                    band_power = psd[mask].mean() if np.any(mask) else 0
                    features.append(band_power)
            return np.array(features)
        
        feature_list = [extract_features(win_acc_data[i]) for i in range(win_acc_data.shape[0])]
        window_features = np.stack(feature_list)
    else:
        window_features = None
    
    # ============================================================
    # FILTER TO VALIDATION SUBJECTS
    # ============================================================
    subjects = np.array([str(s) for s in win_subjects.reshape(-1)])
    unique_subjects = np.unique(subjects)
    
    # Check which validation subjects exist
    available_val_subjects = [s for s in validation_subjects if s in unique_subjects]
    missing_val_subjects = [s for s in validation_subjects if s not in unique_subjects]
    
    print(f"\nSubject filtering:")
    print(f"  • Requested validation subjects: {validation_subjects}")
    print(f"  • Available in data: {available_val_subjects}")
    if missing_val_subjects:
        print(f"  ⚠️  Not found in data: {missing_val_subjects}")
    print(f"  • All available subjects: {list(unique_subjects)}")
    
    if not available_val_subjects:
        print(f"\n❌ ERROR: None of the requested subjects found in data!")
        return
    
    # Filter to validation subjects only
    val_mask = np.isin(subjects, available_val_subjects)
    val_idx = np.where(val_mask)[0]
    
    X_val = win_acc_data[val_idx]
    y_val = win_chorea[val_idx]
    y_val_original = win_chorea_original[val_idx]
    val_subjects_arr = subjects[val_idx]
    
    print(f"\nData for validation:")
    print(f"  • Number of windows: {len(X_val)}")
    print(f"  • Label distribution:")
    for label in label_list:
        count = (y_val == label).sum()
        print(f"      Label {label}: {count} samples")
    
    # Create mask and prepare data
    mask_val = (y_val >= 0).astype(float)
    y_val = np.maximum(y_val, 0)
    
    feats_val = window_features[val_idx] if window_features is not None else None
    
    # Dataset class
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
    
    val_loader = DataLoader(
        ChoreaDataset(X_val, y_val, mask_val, features=feats_val),
        batch_size=64, shuffle=False
    )
    
    # ============================================================
    # LOAD MODEL AND RUN INFERENCE
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Loading model and running inference...")
    print(f"{'='*60}")
    
    # Create model with same configuration
    model = get_sslnet(
        tag='v1.0.0', 
        pretrained=False,
        num_classes=num_classes, 
        model_type='segmentation',
        padding_type='triple_wind', 
        feat_dim=window_features.shape[1] if model_use_features and window_features is not None else 0
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Run inference
    all_preds, all_labels, all_masks, all_probs = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            if model_use_features:
                X_batch, y_batch, mask, feats = batch
                feats = feats.to(device)
            else:
                X_batch, y_batch, mask = batch
                feats = None
            
            X_batch = X_batch.to(device)
            logits = model(X_batch, feats).cpu()
            
            # Different prediction methods based on loss type
            if model_use_ordinal_loss:
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
    y_true_original = y_val_original.reshape(-1)[valid_mask.numpy()]
    val_subjects_per_sample = np.repeat(val_subjects_arr, y_val.shape[1])[valid_mask.numpy()]
    
    # ============================================================
    # COMPUTE AND DISPLAY RESULTS
    # ============================================================
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  • Total valid samples: {len(y_true)}")
    print(f"  • Accuracy:  {acc:.3f}")
    print(f"  • Precision: {prec:.3f}")
    print(f"  • Recall:    {rec:.3f}")
    print(f"  • F1-Score:  {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    total_sum = np.sum(cm)
    
    display_labels = label_list + ['Total']
    n_classes = len(label_list)
    
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
    
    loss_name = "Ordinal" if model_use_ordinal_loss else "MaskedCE"
    metrics_str = f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}"
    ax.set_title(f"Validation Confusion Matrix - {loss_name}\n"
                 f"Subjects: {', '.join(available_val_subjects)}\n{metrics_str}", fontsize=11)
    
    plt.tight_layout()
    
    # Save confusion matrix
    val_output_dir = os.path.join(VIZUALIZE_DIR, 'validation_results')
    os.makedirs(val_output_dir, exist_ok=True)
    
    subjects_str_short = '_'.join(available_val_subjects[:3])
    if len(available_val_subjects) > 3:
        subjects_str_short += f'_and_{len(available_val_subjects)-3}_more'
    
    conf_matrix_path = os.path.join(val_output_dir, f"validation_confusion_matrix_{subjects_str_short}.png")
    plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved confusion matrix to: {conf_matrix_path}")
    
    # Per-subject analysis
    print(f"\n{'='*60}")
    print(f"PER-SUBJECT RESULTS")
    print(f"{'='*60}")
    
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
        
        pred_counts = [np.sum(subject_preds == i) for i in label_list]
        true_counts = [np.sum(subject_true == i) for i in label_list]
        
        x = np.arange(len(label_list))
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
        ax.set_xticklabels(label_list)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        subject_acc = accuracy_score(subject_true, subject_preds)
        ax.text(0.02, 0.98, f'Acc: {subject_acc:.2f}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        subject_f1 = f1_score(subject_true, subject_preds, average='macro', zero_division=0)
        print(f"  Subject {subject}: {len(subject_true)} samples, Accuracy: {subject_acc:.3f}, F1: {subject_f1:.3f}")
    
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Per-Subject Label Distribution (True vs Predicted)', fontsize=14, y=1.00)
    plt.tight_layout()
    
    subject_hist_path = os.path.join(val_output_dir, f"validation_per_subject_{subjects_str_short}.png")
    plt.savefig(subject_hist_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved per-subject histogram to: {subject_hist_path}")
    
    # Classification report
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=label_list, zero_division=0))
    
    # Error analysis for combined labels
    if model_use_combined_labels:
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS (1→0 errors by original label)")
        print(f"{'='*60}")
        
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
            
            print(f"  Total 1→0 errors: {total_errors}")
            print(f"  Originally label 1: {orig_1_errors} (error rate: {error_rate_1:.1f}%)")
            print(f"  Originally label 2: {orig_2_errors} (error rate: {error_rate_2:.1f}%)")
        else:
            print(f"  No 1→0 errors found!")
    
    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {val_output_dir}")
    print(f"{'='*60}\n")


def train_on_all_subjects_function(win_acc_data, win_chorea, win_subjects, window_features, num_label_classes, label_list, device, wandb_initialized):
    """
    Train the model on all subjects using the configured loss function.
    Saves the final trained model and weights.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ON ALL SUBJECTS")
    print(f"{'='*60}")
    
    subjects = np.array([str(s) for s in win_subjects.reshape(-1)])
    unique_subjects = np.unique(subjects)
    unique_subjects.sort()
    
    print(f"\nTraining on all {len(unique_subjects)} subjects:")
    print(f"  {', '.join(sorted(unique_subjects))}")
    print(f"{'='*60}\n")
    
    # Define ChoreaDataset class
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
    
    # Use all data for training
    X_train = win_acc_data
    y_train = win_chorea
    
    mask_train = (y_train >= 0).astype(float)
    y_train = np.maximum(y_train, 0)
    feats_train = window_features if window_features is not None else None
    
    # Determine number of classes and loss type based on configuration
    if use_ordinal_loss:
        num_classes = num_label_classes - 1  # Ordinal loss: K-1 outputs for K levels
        loss_name = "Ordinal"
        print(f"Using Cumulative Ordinal Loss with {num_classes} outputs for {num_label_classes} classes")
    else:
        num_classes = num_label_classes  # Masked CE: K outputs (one per class)
        loss_name = "MaskedCE"
        print(f"Using Masked Cross-Entropy Loss with {num_classes} outputs for {num_label_classes} classes")
    
    # Compute class weights to handle imbalanced data
    valid_train_labels = y_train[mask_train > 0]
    unique_classes, class_counts = np.unique(valid_train_labels, return_counts=True)
    
    print(f"\nClass Distribution (valid samples only):")
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
            print(f"  Class {int(cls)}: {count:5d} samples ({count/total_samples*100:5.1f}%) → weight: {weight:.3f}")
    else:
        class_weights_tensor = None
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {int(cls)}: {count:5d} samples ({count/total_samples*100:5.1f}%)")
    
    # Create DataLoader
    train_loader = DataLoader(
        ChoreaDataset(X_train, y_train, mask_train, features=feats_train),
        batch_size=64, shuffle=True
    )
    
    # === Model setup ===
    ssl_checkpoint_path = os.path.join(SSL_OUTPUT_DIR, 'ssl_model_full_dataset_best.pth')
    
    if use_ssl_encoder and os.path.exists(ssl_checkpoint_path):
        print(f"\n🔄 Loading SSL-trained encoder from full dataset...")
        model = get_sslnet(tag='v1.0.0', pretrained=False,
                          num_classes=num_classes, model_type='segmentation',
                          padding_type='triple_wind', feat_dim=window_features.shape[1] if use_features else 0)
        
        checkpoint = torch.load(ssl_checkpoint_path, map_location=device)
        if 'feature_extractor_state_dict' in checkpoint:
            model.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            val_acc = checkpoint.get('val_acc', 0) * 100
            epoch = checkpoint.get('epoch', 'N/A')
            print(f"  ✓ SSL encoder loaded (Epoch: {epoch}, Val Acc: {val_acc:.2f}%)")
        else:
            print(f"  ⚠️  Could not find feature_extractor_state_dict in checkpoint")
            print(f"  → Using original pretrained weights instead")
            model = get_sslnet(tag='v1.0.0', pretrained=True,
                              num_classes=num_classes, model_type='segmentation',
                              padding_type='triple_wind', feat_dim=window_features.shape[1] if use_features else 0)
    else:
        if use_ssl_encoder:
            print(f"\n  ⚠️  SSL checkpoint not found at {ssl_checkpoint_path}")
            print(f"  → Using original pretrained weights instead")
        model = get_sslnet(tag='v1.0.0', pretrained=True,
                          num_classes=num_classes, model_type='segmentation',
                          padding_type='triple_wind', feat_dim=window_features.shape[1] if use_features else 0)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss function classes
    class MaskedCrossEntropyLoss(torch.nn.Module):
        def __init__(self, class_weights=None):
            super().__init__()
            self.ce = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        def forward(self, input, target, mask):
            loss = self.ce(input, target)
            masked_loss = loss * mask
            return masked_loss.sum() / (mask.sum() + 1e-6)
    
    class MaskedFocalLoss(torch.nn.Module):
        """
        Focal Loss for handling class imbalance by down-weighting easy examples.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.gamma = gamma
            self.class_weights = class_weights
            self.reduction = reduction
            
        def forward(self, input, target, mask):
            # input: [B, num_classes, T], target: [B, T], mask: [B, T]
            B, C, T = input.shape
            
            # Ensure target is long type and clamp to valid range [0, C-1]
            target = target.long()
            target = torch.clamp(target, 0, C - 1)
            
            # Reshape for cross_entropy: input [B*T, C], target [B*T]
            input_2d = input.permute(0, 2, 1).contiguous().view(-1, C)  # [B*T, C]
            target_1d = target.contiguous().view(-1)  # [B*T]
            
            # Compute cross entropy with class weights
            if self.class_weights is not None:
                # Ensure class weights are on the same device as input
                class_weights_device = self.class_weights.to(input.device)
                ce_loss = F.cross_entropy(input_2d, target_1d, reduction='none', weight=class_weights_device)
            else:
                ce_loss = F.cross_entropy(input_2d, target_1d, reduction='none')
            
            # Reshape back to [B, T]
            ce_loss = ce_loss.view(B, T)
            
            # Compute softmax probabilities
            probs = F.softmax(input, dim=1)  # [B, C, T]
            
            # Get probabilities of true class
            target_one_hot = F.one_hot(target, num_classes=C).permute(0, 2, 1).float()  # [B, C, T]
            pt = (probs * target_one_hot).sum(dim=1)  # [B, T]
            
            # Compute focal term: (1 - p_t)^gamma
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply focal weight and mask
            focal_loss = focal_weight * ce_loss * mask
            
            return focal_loss.sum() / (mask.sum() + 1e-6)

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
                # Create weight tensor based on true labels
                sample_weights = torch.ones_like(mask)
                for c in range(len(self.class_weights)):
                    sample_weights[labels == c] = self.class_weights[c]
                loss = loss * sample_weights.unsqueeze(1)
            
            return loss.sum() / (mask.sum() * K + 1e-8)
    
    # Select loss function based on configuration
    if use_ordinal_loss:
        criterion = CumulativeOrdinalLoss(class_weights=class_weights_tensor)
        loss_desc = "Ordinal Loss"
        if use_class_weights:
            loss_desc += " with class weights"
        print(f"\nUsing {loss_desc}")
    else:
        if use_focal_loss:
            criterion = MaskedFocalLoss(class_weights=class_weights_tensor, gamma=2.0)
            loss_desc = "Focal Loss (gamma=2.0)"
        else:
            criterion = MaskedCrossEntropyLoss(class_weights=class_weights_tensor)
            loss_desc = "Masked CE Loss"
        if use_class_weights:
            loss_desc += " with class weights"
        print(f"Using {loss_desc}")
    
    # === Training ===
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    if use_early_stopping:
        print(f"Training config: max_epochs={max_epochs}, early_stopping_patience={early_stopping_patience}, min_epochs={min_epochs}")
        print(f"Note: Early stopping uses training loss since there's no validation set")
    else:
        print(f"Training config: max_epochs={max_epochs} (early stopping disabled)")
    print(f"{'='*60}\n")
    
    best_train_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
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
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Check for improvement (using training loss since no validation set)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improvement_marker = " ✓ (best)"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""
        
        print(f"Epoch {epoch + 1:3d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | LR: {learning_rate:.2e}{improvement_marker}")
        
        # Log to wandb
        if wandb_initialized:
            log_dict = {
                "train_loss": avg_train_loss,
                "learning_rate": learning_rate,
                "epoch": epoch + 1,
            }
            if avg_train_loss < best_train_loss:
                log_dict["best_train_loss"] = best_train_loss
            wandb.log(log_dict, step=epoch + 1)
        
        # Early stopping check (using training loss)
        if use_early_stopping and epoch >= min_epochs - 1 and epochs_without_improvement >= early_stopping_patience:
            print(f"\n  ⚡ Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping_patience} epochs)")
            if wandb_initialized:
                wandb.log({"early_stopped": True, "epochs_trained": epoch + 1}, step=epoch + 1)
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  ✓ Loaded best model (train_loss={best_train_loss:.4f})")
    
    # === Save the trained model ===
    model_save_dir = os.path.join(VIZUALIZE_DIR, 'saved_models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_save_dir, 'model_all_subjects.pth')
    
    # Save model with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_subjects': list(unique_subjects),
        'num_classes': num_classes,
        'num_label_classes': num_label_classes,
        'use_ordinal_loss': use_ordinal_loss,
        'use_combined_labels': use_combined_labels,
        'use_features': use_features,
        'use_class_weights': use_class_weights,
        'window_size': WINDOW_SIZE,
        'best_train_loss': best_train_loss,
        'epochs_trained': epoch + 1,
    }
    torch.save(checkpoint, model_save_path)
    print(f"\n  ✓ Model saved to: {model_save_path}")
    
    # Also save just the weights
    weights_save_path = os.path.join(model_save_dir, 'model_all_subjects_weights.pth')
    torch.save(model.state_dict(), weights_save_path)
    print(f"  ✓ Weights saved to: {weights_save_path}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {model_save_path}")
    print(f"Weights saved to: {weights_save_path}")
    print(f"Loss function used: {loss_desc}")
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"Total epochs trained: {epoch + 1}")
    print(f"{'='*60}\n")
    
    return model, checkpoint


def main():
    # ============================================================
    # INITIALIZE WEIGHTS & BIASES
    # ============================================================
    wandb_initialized = False
    if use_wandb and WANDB_AVAILABLE:
        try:
            # Try to get API key from multiple sources
            api_key = wandb_api_key
            if api_key is None:
                # Check environment variable first
                api_key = os.environ.get('WANDB_API_KEY')
                if api_key is None:
                    # Try to read from .netrc file (common location for wandb)
                    try:
                        import netrc
                        from pathlib import Path
                        netrc_path = Path.home() / '.netrc'
                        if netrc_path.exists():
                            nrc = netrc.netrc(netrc_path)
                            # Try common wandb hostnames
                            for host in ['api.wandb.ai', 'wandb.ai']:
                                try:
                                    _, _, password = nrc.authenticators(host)
                                    if password:
                                        api_key = password
                                        break
                                except (TypeError, KeyError):
                                    continue
                    except Exception:
                        pass
                    
                if api_key is None:
                    # Try to read from wandb settings file
                    try:
                        from pathlib import Path
                        wandb_settings_path = Path.home() / '.config' / 'wandb' / 'settings'
                        if wandb_settings_path.exists():
                            import configparser
                            config = configparser.ConfigParser()
                            config.read(wandb_settings_path)
                            if 'default' in config and 'api_key' in config['default']:
                                api_key = config['default']['api_key']
                    except Exception:
                        pass
            
            # Set API key if found - this is required for wandb to work
            if api_key is not None:
                # Set environment variable first (equivalent to: export WANDB_API_KEY="...")
                os.environ['WANDB_API_KEY'] = api_key
                print(f"\n{'='*60}")
                print(f"✓ Wandb API key set in environment (export WANDB_API_KEY)")
                print(f"{'='*60}\n")
                
                # Run wandb login command - this is required for wandb to connect
                # The command will use WANDB_API_KEY from environment
                try:
                    import subprocess
                    # Run: wandb login (it will use WANDB_API_KEY from environment)
                    result = subprocess.run(
                        ['wandb', 'login'],
                        env=os.environ.copy(),  # Pass environment with WANDB_API_KEY
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        print(f"✓ Wandb login successful (via 'wandb login' command)")
                    else:
                        # If interactive login fails, try with --relogin flag
                        print(f"⚠️  First login attempt returned: {result.returncode}")
                        print(f"   Trying with --relogin flag...")
                        result2 = subprocess.run(
                            ['wandb', 'login', '--relogin'],
                            env=os.environ.copy(),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result2.returncode == 0:
                            print(f"✓ Wandb login successful (via 'wandb login --relogin')")
                        else:
                            print(f"⚠️  Wandb login command failed")
                            print(f"   Output: {result2.stdout}")
                            print(f"   Error: {result2.stderr}")
                            print(f"   Will try to proceed anyway...")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Wandb login command timed out")
                    print(f"   Will try to proceed anyway...")
                except FileNotFoundError:
                    print(f"⚠️  'wandb' command not found in PATH")
                    print(f"   Trying Python API instead...")
                    try:
                        wandb.login(key=api_key, relogin=True)
                        print(f"✓ Wandb login successful (via Python API)")
                    except Exception as login_error:
                        print(f"⚠️  Wandb login failed: {login_error}")
                        print(f"   Will try to proceed anyway...")
                except Exception as cmd_error:
                    print(f"⚠️  Error running wandb login: {cmd_error}")
                    print(f"   Trying Python API instead...")
                    try:
                        wandb.login(key=api_key, relogin=True)
                        print(f"✓ Wandb login successful (via Python API)")
                    except Exception as login_error:
                        print(f"⚠️  Wandb login failed: {login_error}")
                        print(f"   Will try to proceed anyway...")
            else:
                print(f"\n{'='*60}")
                print(f"⚠️  No Wandb API key found")
                print(f"   Set wandb_api_key in configuration or run: export WANDB_API_KEY='your_key'")
                print(f"{'='*60}\n")
            
            # Set offline mode if requested
            if wandb_offline:
                os.environ['WANDB_MODE'] = 'offline'
                print(f"\n{'='*60}")
                print(f"⚠️  Wandb running in OFFLINE mode")
                print(f"   Run 'wandb sync' later to upload results")
                print(f"{'='*60}\n")
            
            # Generate run name if not provided
            if wandb_run_name is None:
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
                if use_focal_loss:
                    run_name_parts.append("focal")
                if train_on_all_subjects:
                    run_name_parts.append("all_subjects")
                elif use_leave_one_out:
                    run_name_parts.append("loso")
                elif not use_random_split:
                    run_name_parts.append(f"{n_folds}fold")
                else:
                    run_name_parts.append(f"{n_folds}fold_random")
                run_name = "_".join(run_name_parts)
            else:
                run_name = wandb_run_name
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                tags=wandb_tags,
                config={
                    "use_features": use_features,
                    "use_ordinal_loss": use_ordinal_loss,
                    "use_ssl_encoder": use_ssl_encoder,
                    "use_class_weights": use_class_weights,
                    "use_focal_loss": use_focal_loss,
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
            wandb_initialized = True
            print(f"\n{'='*60}")
            print(f"🔮 Weights & Biases initialized")
            print(f"   Project: {wandb_project}")
            print(f"   Run: {run_name}")
            print(f"{'='*60}\n")
        except Exception as e:
            error_msg = str(e)
            print(f"\n{'='*60}")
            print(f"⚠️  Failed to initialize Weights & Biases")
            if "not logged in" in error_msg or "401" in error_msg or "Unauthorized" in error_msg:
                print(f"   Authentication error: You need to log in to wandb first.")
                print(f"   Run this command in your terminal:")
                print(f"      wandb login")
                print(f"   Or set WANDB_API_KEY environment variable")
            else:
                print(f"   Error: {error_msg}")
            print(f"   Continuing without wandb logging...")
            print(f"{'='*60}\n")
            wandb_initialized = False
    elif use_wandb and not WANDB_AVAILABLE:
        print(f"\n⚠️  Wandb requested but not available. Install with: pip install wandb\n")
        wandb_initialized = False
    else:
        wandb_initialized = False
    
    if preprocessing_mode:
        win_data_all_sub = np.empty((0, 3, WINDOW_SIZE)) 
        win_labels_all_sub = win_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0, WINDOW_SIZE))
        win_video_time_all_sub = np.empty((0,1))
        NumWin = []

        for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
            try:
                data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
            except:
                print(f"Can't open the file {file}")
                continue

            try:
                acc_data = data_file['acc_data'].astype('float')
            except:
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
                except:
                    print(f"Failed to process acc_data in {file}")
                    continue

            labels = data_file.get('label_data', None)
            chorea = data_file.get('chorea_labels', None)
            video_time = data_file.get('time_data', None)
            subject_name = file.split('.')[0]

            acc_data = preprocessing.bandpass_filter(data=acc_data, low_cut=0.2, high_cut=15,
                                                     sampling_rate=SRC_SAMPLE_RATE, order=4)

            acc_data, labels, chorea, video_time = preprocessing.resample(
                data=acc_data, labels=labels, chorea=chorea, video_time=video_time,
                original_fs=SRC_SAMPLE_RATE, target_fs=30)

            data, labels, chorea, video_time, shift, NumWinSub = preprocessing.data_windowing(
                data=acc_data, labels=labels, chorea=chorea, video_time=video_time,
                window_size=WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
                std_th=STD_THRESH, model_type='segmentation', subject=subject_name)

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

        np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_new_subjects.npz'), **res)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    print("start loading input file")
    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_new_subjects.npz'))
    # input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_hd_only_segmentation_triple_wind_no_shift.npz'))
    print("done loading input file")

    # Load reference file to get valid subjects
    reference_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_hd_only_segmentation_triple_wind_no_shift.npz'))
    reference_subjects = reference_file['win_subjects']
    reference_subjects_str = np.array([str(s) for s in reference_subjects.reshape(-1)])
    valid_subjects_set = set(np.unique(reference_subjects_str))
    reference_file.close()
    
    print(f"\n{'='*60}")
    print(f"Subject Filtering (based on reference file)")
    print(f"{'='*60}")
    print(f"Reference file: windows_input_to_multiclass_model_hd_only_segmentation_triple_wind_no_shift.npz")
    print(f"Valid subjects in reference: {len(valid_subjects_set)}")
    print(f"  {', '.join(sorted(valid_subjects_set))}")

    win_acc_data = input_file['win_data_all_sub']
    win_acc_data = np.transpose(win_acc_data, [0, 2, 1])
    win_labels = input_file['win_labels_all_sub']
    win_chorea = input_file['win_chorea_all_sub']
    win_subjects = input_file['win_subjects']
    win_shift = input_file['win_shift_all_sub']
    win_shift = np.mean(win_shift, axis=-1)

    left = win_acc_data[0:-2:2]
    mid = win_acc_data[1:-1:2]
    right = win_acc_data[2::2]
    mid_labels = win_labels[1:-1:2]
    mid_chorea = win_chorea[1:-1:2]

    # Only keep if middle window contains any walking & at least 2 valid chorea samples
    walking_mask = (mid_labels > 0).sum(axis=1) > 0
    valid_chorea_mask = (mid_chorea >= 0).sum(axis=1) >= 2
    final_mask = walking_mask & valid_chorea_mask
    
    print(f"\n{'='*60}")
    print(f"Data Filtering Statistics:")
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

    # ============================================================
    # FILTER SUBJECTS BASED ON REFERENCE FILE
    # ============================================================
    subjects_str = np.array([str(s) for s in win_subjects.reshape(-1)])
    subject_mask = np.array([s in valid_subjects_set for s in subjects_str])
    
    subjects_before = len(np.unique(subjects_str))
    subjects_after = len(np.unique(subjects_str[subject_mask]))
    
    print(f"\n{'='*60}")
    print(f"Filtering Subjects (keep only those in reference file)")
    print(f"{'='*60}")
    print(f"Subjects before filtering: {subjects_before}")
    print(f"Subjects after filtering:  {subjects_after}")
    
    if subjects_before > subjects_after:
        removed_subjects = set(np.unique(subjects_str)) - valid_subjects_set
        print(f"Removed subjects: {', '.join(sorted(removed_subjects))}")
        print(f"Windows before filtering: {len(left_wind)}")
    
    # Apply subject filter
    left_wind = left_wind[subject_mask]
    mid_wind = mid_wind[subject_mask]
    right_wind = right_wind[subject_mask]
    mid_chorea = mid_chorea[subject_mask]
    win_subjects = win_subjects[subject_mask]
    
    print(f"Windows after filtering:  {len(left_wind)}")
    print(f"Windows removed:           {len(subject_mask) - subject_mask.sum()}")
    print(f"{'='*60}\n")

    # ============================================================
    # REMOVE HEALTHY SUBJECTS (subjects with 'CO' in their ID)
    # ============================================================
    
    # Identify healthy subjects (those containing 'CO' in their ID)
    subjects_str = np.array([str(s) for s in win_subjects])
    healthy_mask = np.array(['CO' in s for s in subjects_str])
    hd_mask = ~healthy_mask  # HD patients (not healthy)
    
    # Get unique subjects before filtering
    unique_before = np.unique(subjects_str)
    healthy_subjects = np.unique(subjects_str[healthy_mask])
    hd_subjects = np.unique(subjects_str[hd_mask])
    
    print(f"\n{'='*60}")
    print(f"Removing Healthy Subjects (ID contains 'CO'):")
    print(f"{'='*60}")
    print(f"Total subjects before filtering: {len(unique_before)}")
    print(f"Healthy subjects (removed): {len(healthy_subjects)}")
    if len(healthy_subjects) > 0:
        print(f"  → {', '.join(healthy_subjects)}")
    print(f"HD subjects (kept): {len(hd_subjects)}")
    print(f"  → {', '.join(hd_subjects)}")
    print(f"\nWindows before filtering: {len(left_wind)}")
    print(f"Windows removed (healthy): {healthy_mask.sum()}")
    print(f"Windows kept (HD): {hd_mask.sum()}")
    print(f"{'='*60}\n")
    
    # Apply the filter to keep only HD subjects
    left_wind = left_wind[hd_mask]
    mid_wind = mid_wind[hd_mask]
    right_wind = right_wind[hd_mask]
    mid_chorea = mid_chorea[hd_mask]
    win_subjects = win_subjects[hd_mask]

    win_chorea = mid_chorea
    
    # Keep original labels for error analysis
    win_chorea_original = win_chorea.copy()
    
    # Determine number of classes based on configuration
    if use_combined_labels:
        # Remap chorea labels to 3 classes: 0->0, 1,2->1, 3,4->2
        print(f"\n{'='*60}")
        print(f"Remapping Chorea Labels to 3 Classes:")
        print(f"{'='*60}")
        print(f"Original label distribution:")
        for label in range(5):
            count = (win_chorea == label).sum()
            print(f"  Label {label}: {count} samples")
        
        # Apply remapping
        win_chorea_remapped = win_chorea.copy()
        win_chorea_remapped[win_chorea == 1] = 1
        win_chorea_remapped[win_chorea == 2] = 1
        win_chorea_remapped[win_chorea == 3] = 2
        win_chorea_remapped[win_chorea == 4] = 2
        win_chorea = win_chorea_remapped
        
        print(f"\nRemapped label distribution:")
        for label in range(3):
            count = (win_chorea == label).sum()
            print(f"  Label {label}: {count} samples")
        print(f"{'='*60}\n")
        
        num_label_classes = 3
        label_list = [0, 1, 2]
    else:
        # Keep original 5 classes (0-4)
        print(f"\n{'='*60}")
        print(f"Using Original 5 Chorea Labels (0-4):")
        print(f"{'='*60}")
        print(f"Label distribution:")
        for label in range(5):
            count = (win_chorea == label).sum()
            print(f"  Label {label}: {count} samples")
        print(f"{'='*60}\n")
        
        num_label_classes = 5
        label_list = [0, 1, 2, 3, 4]
    
    # Stack windows
    win_acc_data = np.stack([left_wind, mid_wind, right_wind], axis=2).reshape(-1, 3, WINDOW_SIZE * 3)

    from scipy.stats import skew, kurtosis, entropy
    from scipy.fft import rfft, rfftfreq

    def extract_features(acc_window, fs=30):
        """
        acc_window: np.array shape [3, WINDOW_SIZE]
        fs: sampling frequency (Hz)
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


    # --- Apply feature extraction to all windows ---
    if use_features:
        feature_list = [extract_features(win_acc_data[i]) for i in range(win_acc_data.shape[0])]
        window_features = np.stack(feature_list)  # shape: [num_windows, num_features]
        print("Window features shape:", window_features.shape)
    else:
        window_features = None  

    # === Dataloader preparation ===
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
    
    class MaskedFocalLoss(torch.nn.Module):
        """
        Focal Loss for handling class imbalance by down-weighting easy examples.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.gamma = gamma
            self.class_weights = class_weights
            self.reduction = reduction
            
        def forward(self, input, target, mask):
            # input: [B, num_classes, T], target: [B, T], mask: [B, T]
            B, C, T = input.shape
            
            # Ensure target is long type and clamp to valid range [0, C-1]
            target = target.long()
            target = torch.clamp(target, 0, C - 1)
            
            # Reshape for cross_entropy: input [B*T, C], target [B*T]
            input_2d = input.permute(0, 2, 1).contiguous().view(-1, C)  # [B*T, C]
            target_1d = target.contiguous().view(-1)  # [B*T]
            
            # Compute cross entropy with class weights
            if self.class_weights is not None:
                # Ensure class weights are on the same device as input
                class_weights_device = self.class_weights.to(input.device)
                ce_loss = F.cross_entropy(input_2d, target_1d, reduction='none', weight=class_weights_device)
            else:
                ce_loss = F.cross_entropy(input_2d, target_1d, reduction='none')
            
            # Reshape back to [B, T]
            ce_loss = ce_loss.view(B, T)
            
            # Compute softmax probabilities
            probs = F.softmax(input, dim=1)  # [B, C, T]
            
            # Get probabilities of true class
            target_one_hot = F.one_hot(target, num_classes=C).permute(0, 2, 1).float()  # [B, C, T]
            pt = (probs * target_one_hot).sum(dim=1)  # [B, T]
            
            # Compute focal term: (1 - p_t)^gamma
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply focal weight and mask
            focal_loss = focal_weight * ce_loss * mask
            
            return focal_loss.sum() / (mask.sum() + 1e-6)

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
                # Create weight tensor based on true labels
                sample_weights = torch.ones_like(mask)
                for c in range(len(self.class_weights)):
                    sample_weights[labels == c] = self.class_weights[c]
                loss = loss * sample_weights.unsqueeze(1)
            
            return loss.sum() / (mask.sum() * K + 1e-8)
        
    # ------------------ Axis normalization ------------------
    # def normalize_axes(X):
    #     """
    #     X: [num_samples, 3, num_timesteps]
    #     Returns X_rot: rotated axes where:
    #     - first axis = main movement (PCA)
    #     - second axis = orthogonal
    #     - third axis = gravity aligned
    #     """
    #     X_rot = np.zeros_like(X)
    #     for i in range(X.shape[0]):
    #         window = X[i].T  # [timesteps, 3]

    #         # 1. Gravity vector
    #         g = window.mean(axis=0)
    #         g = g / (np.linalg.norm(g) + 1e-8)

    #         # 2. Remove gravity
    #         window_no_g = window - g * np.dot(window, g[:, None])

    #         # 3. PCA on remaining movement
    #         cov = np.cov(window_no_g, rowvar=False)
    #         eigvals, eigvecs = np.linalg.eigh(cov)
    #         order = np.argsort(eigvals)[::-1]
    #         R = eigvecs[:, order]

    #         # 4. Rotate window
    #         X_rot[i] = (window @ R).T
    #     return X_rot

        # ------------------------------
    # GroupKFold subject-wise CV
    # ------------------------------

    # --- Axis normalization
    # win_acc_data = normalize_axes(win_acc_data)

    subjects = np.array([str(s) for s in win_subjects.reshape(-1)])
    unique_subjects = np.unique(subjects)
    unique_subjects.sort()  # Ensure deterministic order
    subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
    groups = np.array([subject_to_idx[s] for s in subjects])
    
    # ============================================================
    # FOLD CONFIGURATION - Leave-One-Out, Random, or Custom
    # ============================================================
    
    if use_leave_one_out:
        # Leave-One-Subject-Out (LOSO) cross-validation
        # Each fold leaves out exactly one subject for validation
        from sklearn.model_selection import LeaveOneGroupOut
        
        logo = LeaveOneGroupOut()
        
        # Generate fold_to_val_subjects from LeaveOneGroupOut
        fold_to_val_subjects = {}
        for fold_num, (train_idx, val_idx) in enumerate(logo.split(win_acc_data, win_chorea, groups), 1):
            val_subjects_in_fold = np.unique(subjects[val_idx])
            fold_to_val_subjects[fold_num] = list(val_subjects_in_fold)
        
        n_splits = len(fold_to_val_subjects)
        print(f"\n{'='*60}")
        print(f"Using LEAVE-ONE-SUBJECT-OUT (LOSO) Cross-Validation")
        print(f"Number of folds: {n_splits} (one per subject)")
        print(f"Available subjects: {', '.join(sorted(unique_subjects))}")
        print(f"\nFold assignments (one subject per fold):")
        for fold_num, val_subs in fold_to_val_subjects.items():
            print(f"  Fold {fold_num}: validation = {val_subs[0]}")
        print(f"{'='*60}\n")
    elif use_random_split:
        # Random GroupKFold split
        from sklearn.model_selection import GroupKFold
        
        np.random.seed(random_seed)
        
        # Shuffle subjects before splitting for randomness
        shuffled_subjects = unique_subjects.copy()
        np.random.shuffle(shuffled_subjects)
        
        # Create a mapping from original subject order to shuffled order
        subject_to_shuffled_idx = {subj: i for i, subj in enumerate(shuffled_subjects)}
        shuffled_groups = np.array([subject_to_shuffled_idx[s] for s in subjects])
        
        gkf = GroupKFold(n_splits=n_folds)
        
        # Generate fold_to_val_subjects from GroupKFold
        fold_to_val_subjects = {}
        for fold_num, (train_idx, val_idx) in enumerate(gkf.split(win_acc_data, win_chorea, shuffled_groups), 1):
            val_subjects_in_fold = np.unique(subjects[val_idx])
            fold_to_val_subjects[fold_num] = list(val_subjects_in_fold)
        
        n_splits = len(fold_to_val_subjects)
        print(f"\n{'='*60}")
        print(f"Using RANDOM GroupKFold Split (seed={random_seed})")
        print(f"Number of folds: {n_splits}")
        print(f"Available subjects: {', '.join(sorted(unique_subjects))}")
        print(f"\nRandom fold assignments:")
        for fold_num, val_subs in fold_to_val_subjects.items():
            print(f"  Fold {fold_num}: validation = {', '.join(sorted(val_subs))}")
        print(f"{'='*60}\n")
    else:
        # Custom/Manual fold configuration
        # Define which subjects go into validation set for each fold
        # Modify this dictionary to control fold assignments
        # Subjects not listed in any fold will only appear in training sets
        
        fold_to_val_subjects = {
            1: ['IW13GHI', 'IW15GHI', 'IW6GHI', 'IW9TC'],  # Fold 1: these subjects in validation
            2: ['IW11TC', 'IW15TC', 'IW3GHI', 'IW5TC'],  # Fold 2: these subjects in validation
            3: ['IW10TC', 'IW12GHI', 'IW13TC', 'IW7TC', 'IW8TC'],  # Fold 3: these subjects in validation
            4: ['IW14TC', 'IW4GHI', 'IW4TC', 'IW5GHI'],  # Fold 4: these subjects in validation
            5: ['IW10GHI', 'IW11GHI', 'IW12TC', 'IW14GHI', 'IW6TC'],  # Fold 5: these subjects in validation
        } 
        
        # Validate that all specified subjects exist in the data
        all_specified_subjects = set()
        for fold_subjects in fold_to_val_subjects.values():
            all_specified_subjects.update(fold_subjects)
        
        missing_subjects = all_specified_subjects - set(unique_subjects)
        if missing_subjects:
            print(f"\n⚠️  WARNING: These subjects are specified in fold_to_val_subjects but not found in data:")
            print(f"    {', '.join(sorted(missing_subjects))}")
            print(f"    Available subjects: {', '.join(sorted(unique_subjects))}")
        
        n_splits = len(fold_to_val_subjects)
        print(f"\n{'='*60}")
        print(f"Using CUSTOM Fold Configuration")
        print(f"Number of folds: {n_splits}")
        print(f"Available subjects: {', '.join(sorted(unique_subjects))}")
        print(f"\nFold assignments:")
        for fold_num, val_subs in fold_to_val_subjects.items():
            print(f"  Fold {fold_num}: validation = {', '.join(val_subs)}")
        print(f"{'='*60}\n")

    # Check if we should train on all subjects
    if train_on_all_subjects:
        print(f"\n{'='*60}")
        print(f"TRAINING ON ALL SUBJECTS MODE")
        print(f"{'='*60}")
        print(f"Skipping cross-validation. Training on all available subjects.")
        print(f"{'='*60}\n")
        
        # Determine number of classes and loss type based on configuration
        if use_ordinal_loss:
            num_classes = num_label_classes - 1  # Ordinal loss: K-1 outputs for K levels
        else:
            num_classes = num_label_classes  # Masked CE: K outputs (one per class)
        
        # Train on all subjects
        model, checkpoint = train_on_all_subjects_function(
            win_acc_data, win_chorea, win_subjects, window_features,
            num_label_classes, label_list, device, wandb_initialized
        )
        
        # Finish wandb run
        if wandb_initialized:
            wandb.finish()
            print(f"🔮 Wandb run completed and synced")
        
        return
    
    # Continue with cross-validation if train_on_all_subjects is False
    fold_results = []
    
    # Lists to collect predictions and labels from all folds for aggregated confusion matrix
    all_folds_y_pred = []
    all_folds_y_true = []
    
    # Track maximum step used during training for proper wandb logging
    max_training_step = 0
    
    # Determine number of classes and loss type based on configuration
    if use_ordinal_loss:
        num_classes = num_label_classes - 1  # Ordinal loss: K-1 outputs for K levels
        loss_name = "Ordinal"
        print(f"\n{'='*60}")
        print(f"Using Cumulative Ordinal Loss with {num_classes} outputs for {num_label_classes} classes")
        print(f"{'='*60}")
    else:
        num_classes = num_label_classes  # Masked CE: K outputs (one per class)
        loss_name = "MaskedCE"
        print(f"\n{'='*60}")
        print(f"Using Masked Cross-Entropy Loss with {num_classes} outputs for {num_label_classes} classes")
        print(f"{'='*60}")

    for fold_num, val_subject_list in fold_to_val_subjects.items():
        print(f"\n=== Fold {fold_num}/{n_splits} ===")
        
        fold_step_offset = max_training_step
        
        # Create masks for validation and training based on specified subjects
        val_mask = np.isin(subjects, val_subject_list)
        train_mask = ~val_mask
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        if len(val_idx) == 0:
            print(f"  ⚠️  WARNING: No validation samples found for subjects {val_subject_list}. Skipping fold.")
            continue

        X_train, y_train = win_acc_data[train_idx], win_chorea[train_idx]
        X_val, y_val = win_acc_data[val_idx], win_chorea[val_idx]
        
        # Get validation subjects
        val_subjects = subjects[val_idx]
        unique_val_subjects = np.unique(val_subjects)
        print(f"\nValidation subjects ({len(unique_val_subjects)}): {', '.join(unique_val_subjects)}")

        mask_train = (y_train >= 0).astype(float)
        mask_val = (y_val >= 0).astype(float)

        y_train = np.maximum(y_train, 0)
        y_val = np.maximum(y_val, 0)

        feats_train = window_features[train_idx] if window_features is not None else None
        feats_val = window_features[val_idx] if window_features is not None else None

        # Compute class weights to handle imbalanced data
        # Only use valid (masked) samples for weight calculation
        valid_train_labels = y_train[mask_train > 0]
        unique_classes, class_counts = np.unique(valid_train_labels, return_counts=True)
        
        print(f"\n  Class Distribution (valid samples only):")
        total_samples = len(valid_train_labels)
        
        if use_class_weights:
            # Initialize weights array for all possible classes
            num_total_classes = num_label_classes
            class_weights_array = np.ones(num_total_classes)
            
            # Compute inverse frequency weights for classes that exist
            class_weights_temp = total_samples / (len(unique_classes) * class_counts)
            
            # Assign weights to the correct class indices
            for cls, weight in zip(unique_classes, class_weights_temp):
                class_weights_array[int(cls)] = weight
            
            # Normalize weights to sum to number of classes for stability
            class_weights_array = class_weights_array * num_total_classes / class_weights_array.sum()
            
            # Convert to tensor
            class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
            
            for cls, count in zip(unique_classes, class_counts):
                weight = class_weights_array[int(cls)]
                print(f"    Class {int(cls)}: {count:5d} samples ({count/total_samples*100:5.1f}%) → weight: {weight:.3f}")
        else:
            class_weights_tensor = None
            for cls, count in zip(unique_classes, class_counts):
                print(f"    Class {int(cls)}: {count:5d} samples ({count/total_samples*100:5.1f}%)")
        
        # Update DataLoaders to include features
        train_loader = DataLoader(
            ChoreaDataset(X_train, y_train, mask_train, features=feats_train),
            batch_size=64, shuffle=True
        )
        val_loader = DataLoader(
            ChoreaDataset(X_val, y_val, mask_val, features=feats_val),
            batch_size=64, shuffle=False
        )

        # === Model setup (fresh each fold) ===
        # Check if SSL-trained encoder is available
        ssl_checkpoint_path = os.path.join(SSL_OUTPUT_DIR, 'ssl_model_full_dataset_best.pth')
        
        if use_ssl_encoder and os.path.exists(ssl_checkpoint_path):
            print(f"\n  🔄 Loading SSL-trained encoder from full dataset...")
            # Create model with fresh weights
            model = get_sslnet(tag='v1.0.0', pretrained=False,
                              num_classes=num_classes, model_type='segmentation',
                              padding_type='triple_wind', feat_dim=window_features.shape[1] if use_features else 0)
            
            # Load SSL-trained feature extractor
            checkpoint = torch.load(ssl_checkpoint_path, map_location=device)
            if 'feature_extractor_state_dict' in checkpoint:
                model.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
                val_acc = checkpoint.get('val_acc', 0) * 100
                epoch = checkpoint.get('epoch', 'N/A')
                print(f"  ✓ SSL encoder loaded (Epoch: {epoch}, Val Acc: {val_acc:.2f}%)")
            else:
                print(f"  ⚠️  Could not find feature_extractor_state_dict in checkpoint")
                print(f"  → Using original pretrained weights instead")
                model = get_sslnet(tag='v1.0.0', pretrained=True,
                                  num_classes=num_classes, model_type='segmentation',
                                  padding_type='triple_wind', feat_dim=window_features.shape[1] if use_features else 0)
        else:
            if use_ssl_encoder:
                print(f"\n  ⚠️  SSL checkpoint not found at {ssl_checkpoint_path}")
                print(f"  → Using original pretrained weights instead")
            # Use original pretrained weights
            model = get_sslnet(tag='v1.0.0', pretrained=True,
                              num_classes=num_classes, model_type='segmentation',
                              padding_type='triple_wind', feat_dim=window_features.shape[1] if use_features else 0)
        
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Select loss function based on configuration
        if class_weights_tensor is not None:
            print(f"  Class weights tensor: shape={class_weights_tensor.shape}, values={class_weights_tensor}")
        
        if use_ordinal_loss:
            criterion = CumulativeOrdinalLoss(class_weights=class_weights_tensor)
            loss_desc = "Ordinal Loss"
            if use_class_weights:
                loss_desc += " with class weights"
            print(f"  Using {loss_desc}")
        else:
            if use_focal_loss:
                criterion = MaskedFocalLoss(class_weights=class_weights_tensor, gamma=2.0)
                loss_desc = "Focal Loss (gamma=2.0)"
            else:
                criterion = MaskedCrossEntropyLoss(class_weights=class_weights_tensor)
                loss_desc = "Masked CE Loss"
            if use_class_weights:
                loss_desc += " with class weights"
            print(f"  Using {loss_desc}")

        # === Training with Early Stopping ===
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        last_epoch_in_fold = 0  # Track the last epoch reached in this fold
        
        if use_early_stopping:
            print(f"\n  Training config: max_epochs={max_epochs}, early_stopping_patience={early_stopping_patience}, min_epochs={min_epochs}")
        else:
            print(f"\n  Training config: max_epochs={max_epochs} (early stopping disabled)")
        
        for epoch in range(max_epochs):
            # --- Training phase ---
            model.train()
            total_train_loss = 0
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
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- Validation phase ---
            model.eval()
            total_val_loss = 0
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
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # --- Check for improvement ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                improvement_marker = " ✓ (best)"
            else:
                epochs_without_improvement += 1
                improvement_marker = ""
            
            print(f"Epoch {epoch + 1:3d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {learning_rate:.2e}{improvement_marker}")
            
            # --- Log to wandb ---
            if wandb_initialized:
                current_step = fold_step_offset + epoch + 1
                last_epoch_in_fold = epoch + 1  # Update last epoch reached
                max_training_step = max(max_training_step, current_step)
                log_dict = {
                    f"fold_{fold_num}/train_loss": avg_train_loss,
                    f"fold_{fold_num}/val_loss": avg_val_loss,
                    f"fold_{fold_num}/learning_rate": learning_rate,
                    f"fold_{fold_num}/epoch": epoch + 1,
                }
                if avg_val_loss < best_val_loss:
                    log_dict[f"fold_{fold_num}/best_val_loss"] = best_val_loss
                wandb.log(log_dict, step=current_step)
            
            # --- Early stopping check ---
            if use_early_stopping and epoch >= min_epochs - 1 and epochs_without_improvement >= early_stopping_patience:
                print(f"\n  ⚡ Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping_patience} epochs)")
                if wandb_initialized:
                    current_step = fold_step_offset + epoch + 1
                    last_epoch_in_fold = epoch + 1  # Update last epoch reached
                    max_training_step = max(max_training_step, current_step)
                    wandb.log({f"fold_{fold_num}/early_stopped": True, f"fold_{fold_num}/epochs_trained": epoch + 1}, step=current_step)
                break
        
        # Update last_epoch_in_fold if training completed all epochs
        if epoch == max_epochs - 1:
            last_epoch_in_fold = max_epochs
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"  ✓ Loaded best model (val_loss={best_val_loss:.4f})")

        # === Save the trained model ===
        model_save_dir = os.path.join(VIZUALIZE_DIR, 'saved_models')
        os.makedirs(model_save_dir, exist_ok=True)
        
        model_save_path = os.path.join(model_save_dir, f'model_fold{fold_num}.pth')
        
        # Save model with metadata for later use
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
            'epochs_trained': epoch + 1,
        }
        torch.save(checkpoint, model_save_path)
        print(f"\n  ✓ Model saved to: {model_save_path}")

        # === Evaluation on validation fold ===
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
                
                # Different prediction methods based on loss type
                if use_ordinal_loss:
                    # Ordinal loss: use sigmoid and count thresholds
                    # Flipped direction: P(Y <= k), so prediction = K - count(probs > 0.5)
                    probs = torch.sigmoid(logits)
                    pred = (probs > 0.5).sum(dim=1)
                else:
                    # Masked CE: use softmax and argmax
                    probs = F.softmax(logits, dim=1)
                    pred = probs.argmax(dim=1)
                
                all_preds.append(pred)
                all_labels.append(y_batch)
                all_masks.append(mask)
                all_probs.append(probs)

        y_pred = torch.cat(all_preds).view(-1).numpy()
        y_true = torch.cat(all_labels).view(-1).numpy()
        all_masks = torch.cat(all_masks, dim=0).view(-1)
        valid_mask = all_masks.bool()

        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        
        # Collect predictions and labels for aggregated confusion matrix
        all_folds_y_pred.append(y_pred)
        all_folds_y_true.append(y_true)
        
        # Get original labels and subjects for error analysis
        y_true_original = win_chorea_original[val_idx].reshape(-1)[valid_mask.numpy()]
        val_subjects_per_sample = np.repeat(val_subjects, y_val.shape[1])[valid_mask.numpy()]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Fold {fold_num} → Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        fold_results.append((fold_num, acc, prec, rec, f1))
        
        # Log fold metrics to wandb
        if wandb_initialized:
            # Use a step right after the last training epoch in this fold
            # This ensures continuous, monotonically increasing steps across all folds
            eval_step = fold_step_offset + last_epoch_in_fold + 1
            max_training_step = max(max_training_step, eval_step)
            wandb.log({
                f"fold_{fold_num}/accuracy": acc,
                f"fold_{fold_num}/precision": prec,
                f"fold_{fold_num}/recall": rec,
                f"fold_{fold_num}/f1_score": f1,
            }, step=eval_step)

        # === Confusion Matrix ===
        labels = label_list
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Calculate marginals
        row_sums = np.sum(cm, axis=1)
        col_sums = np.sum(cm, axis=0)
        total_sum = np.sum(cm)

        # Create display labels with 'Total'
        display_labels = labels + ['Total']
        n_classes = len(labels)

        # Plot confusion matrix with separate handling for margins
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the core confusion matrix (without margins) with colormap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add text annotations for the core matrix
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12)
        
        # Extend axes to accommodate margins
        ax.set_xlim(-0.5, n_classes + 0.5)
        ax.set_ylim(n_classes + 0.5, -0.5)
        
        # Add gray background rectangles for margin cells
        for i in range(n_classes):
            # Right column (row sums)
            rect = plt.Rectangle((n_classes - 0.5, i - 0.5), 1, 1, 
                                 facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(n_classes, i, format(row_sums[i], 'd'),
                   ha="center", va="center", color="black", fontsize=12)
            
            # Bottom row (column sums)
            rect = plt.Rectangle((i - 0.5, n_classes - 0.5), 1, 1,
                                 facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(i, n_classes, format(col_sums[i], 'd'),
                   ha="center", va="center", color="black", fontsize=12)
        
        # Bottom-right corner (total)
        rect = plt.Rectangle((n_classes - 0.5, n_classes - 0.5), 1, 1,
                             facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(n_classes, n_classes, format(total_sum, 'd'),
               ha="center", va="center", color="black", fontsize=12)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_classes + 1))
        ax.set_yticks(np.arange(n_classes + 1))
        ax.set_xticklabels(display_labels, fontsize=11)
        ax.set_yticklabels(display_labels, fontsize=11)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)

        # Add detailed title
        title_parts = [loss_name]
        if use_class_weights:
            title_parts.append("Weighted")
        if use_focal_loss and not use_ordinal_loss:
            title_parts.append("Focal")
        loss_title = " + ".join(title_parts) if len(title_parts) > 1 else title_parts[0]
        
        metrics_str = (
            f"Accuracy: {acc:.3f} | "
            f"Precision: {prec:.3f} | "
            f"Recall: {rec:.3f} | "
            f"F1: {f1:.3f}"
        )
        ax.set_title(f"Confusion Matrix (3-Class: 0,1,2) - {loss_title}\n{metrics_str}", fontsize=12)

        plt.tight_layout()
        conf_matrix_path = VIZUALIZE_DIR + f"conf_matrix_fold{fold_num}.png"
        plt.savefig(conf_matrix_path)
        
        # Log confusion matrix to wandb
        if wandb_initialized:
            # Use a step right after the last training epoch in this fold (after metrics log)
            # This ensures continuous, monotonically increasing steps across all folds
            eval_step = fold_step_offset + last_epoch_in_fold + 2  # +2 to be after the metrics log
            max_training_step = max(max_training_step, eval_step)
            wandb.log({f"fold_{fold_num}/confusion_matrix": wandb.Image(fig)}, step=eval_step)
        
        plt.show()
        
        # ============================================================
        # ERROR ANALYSIS
        # ============================================================
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS - Fold {fold_num}")
        print(f"{'='*60}")
        
        # 1. Analyze 1→0 errors by original label (only for combined labels mode)
        # Find cases where model predicted 0 but true label was 1 (remapped)
        if use_combined_labels:
            errors_1to0_mask = (y_pred == 0) & (y_true == 1)
            
            if errors_1to0_mask.sum() > 0:
                # Get original labels for these errors
                original_labels_of_errors = y_true_original[errors_1to0_mask]
                
                # Count how many were originally 1 vs 2
                orig_1_errors = (original_labels_of_errors == 1).sum()
                orig_2_errors = (original_labels_of_errors == 2).sum()
                total_errors = errors_1to0_mask.sum()
                
                # Calculate total number of original 1s and 2s in validation set
                total_orig_1 = (y_true_original == 1).sum()
                total_orig_2 = (y_true_original == 2).sum()
                
                # Calculate error rates
                error_rate_1 = (orig_1_errors / total_orig_1 * 100) if total_orig_1 > 0 else 0
                error_rate_2 = (orig_2_errors / total_orig_2 * 100) if total_orig_2 > 0 else 0
                
                print(f"\n1→0 Error Analysis:")
                print(f"  Total errors (predicted 0, true 1): {total_errors}")
                print(f"  Originally label 1: {orig_1_errors} ({orig_1_errors/total_errors*100:.1f}% of errors)")
                print(f"    → Error rate: {orig_1_errors}/{total_orig_1} = {error_rate_1:.1f}%")
                print(f"  Originally label 2: {orig_2_errors} ({orig_2_errors/total_errors*100:.1f}% of errors)")
                print(f"    → Error rate: {orig_2_errors}/{total_orig_2} = {error_rate_2:.1f}%")
                
                # Visualize this with dual information
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Left plot: Error counts
                categories = ['Original\nLabel 1', 'Original\nLabel 2']
                counts = [orig_1_errors, orig_2_errors]
                colors = ['#FF6B6B', '#4ECDC4']
                
                bars1 = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                
                # Add count labels on bars
                for bar, count in zip(bars1, counts):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{count}\n({count/total_errors*100:.1f}%)',
                            ha='center', va='bottom', fontsize=11)
                
                ax1.set_ylabel('Number of Errors', fontsize=12)
                ax1.set_title(f'Error Counts\n(Total: {total_errors} errors)', 
                             fontsize=12)
                ax1.set_ylim(0, max(counts) * 1.25)
                ax1.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Right plot: Error rates
                error_rates = [error_rate_1, error_rate_2]
                bars2 = ax2.bar(categories, error_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                
                # Add error rate labels on bars
                for bar, err_count, total_count, rate in zip(bars2, counts, [total_orig_1, total_orig_2], error_rates):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{rate:.1f}%\n({err_count}/{total_count})',
                            ha='center', va='bottom', fontsize=11)
                
                ax2.set_ylabel('Error Rate (%)', fontsize=12)
                ax2.set_title(f'Error Rates\n(Errors / Total samples per label)', 
                             fontsize=12)
                ax2.set_ylim(0, max(error_rates) * 1.25)
                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
                ax2.legend()
                
                fig.suptitle(f'Fold {fold_num}: 1→0 Error Analysis by Original Label', 
                            fontsize=14)
                
                plt.tight_layout()
                error_analysis_path = VIZUALIZE_DIR + f"error_analysis_1to0_fold{fold_num}.png"
                plt.savefig(error_analysis_path)
                plt.show()
                print(f"  Saved error analysis to: {error_analysis_path}")
        
        # 2. Per-subject histograms
        print(f"\n2. Per-Subject Distribution Analysis:")
        
        unique_val_subs = np.unique(val_subjects_per_sample)
        n_subjects = len(unique_val_subs)
        
        # Create subplots for each subject
        n_cols = min(3, n_subjects)
        n_rows = int(np.ceil(n_subjects / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_subjects == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, subject in enumerate(unique_val_subs):
            ax = axes[idx]
            
            # Get data for this subject
            subject_mask = val_subjects_per_sample == subject
            subject_preds = y_pred[subject_mask]
            subject_true = y_true[subject_mask]
            
            # Count predictions and true labels
            labels_range = label_list
            pred_counts = [np.sum(subject_preds == i) for i in labels_range]
            true_counts = [np.sum(subject_true == i) for i in labels_range]
            
            x = np.arange(len(labels_range))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels', 
                          color='#3498db', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, pred_counts, width, label='Predictions',
                          color='#e74c3c', alpha=0.8, edgecolor='black')
            
            # Add count labels
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
            
            # Calculate accuracy for this subject
            subject_acc = accuracy_score(subject_true, subject_preds)
            ax.text(0.02, 0.98, f'Acc: {subject_acc:.2f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
            
            print(f"  Subject {subject}: {len(subject_true)} samples, Accuracy: {subject_acc:.3f}")
        
        # Hide unused subplots
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
    # AGGREGATED CONFUSION MATRIX FROM ALL FOLDS
    # ============================================================
    
    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS FROM ALL {n_splits} FOLDS")
    print(f"{'='*60}")
    
    # Concatenate all predictions and labels from all folds
    all_y_pred = np.concatenate(all_folds_y_pred)
    all_y_true = np.concatenate(all_folds_y_true)
    
    # Calculate overall metrics
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    
    print(f"\nOverall Metrics (aggregated from all folds):")
    print(f"  Total samples: {len(all_y_true)}")
    print(f"  Accuracy:  {overall_acc:.3f}")
    print(f"  Precision: {overall_prec:.3f}")
    print(f"  Recall:    {overall_rec:.3f}")
    print(f"  F1-Score:  {overall_f1:.3f}")
    
    # Per-fold summary
    print(f"\nPer-Fold Results:")
    print(f"{'-'*60}")
    print(f"{'Fold':<10} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print(f"{'-'*60}")
    
    # Extract metrics (skip fold_num which is first element)
    fold_metrics_array = np.array([(acc, prec, rec, f1) for (fold_num, acc, prec, rec, f1) in fold_results])
    for (fold_num, acc, prec, rec, f1) in fold_results:
        print(f"{'Fold ' + str(fold_num):<10} {acc:>12.3f} {prec:>12.3f} {rec:>12.3f} {f1:>12.3f}")
    
    mean_acc = np.mean(fold_metrics_array[:, 0])
    std_acc = np.std(fold_metrics_array[:, 0])
    mean_prec = np.mean(fold_metrics_array[:, 1])
    std_prec = np.std(fold_metrics_array[:, 1])
    mean_rec = np.mean(fold_metrics_array[:, 2])
    std_rec = np.std(fold_metrics_array[:, 2])
    mean_f1 = np.mean(fold_metrics_array[:, 3])
    std_f1 = np.std(fold_metrics_array[:, 3])
    
    # Log per-fold summary table to wandb
    if wandb_initialized:
        fold_table_data = []
        for (fold_num, acc, prec, rec, f1) in fold_results:
            fold_table_data.append([fold_num, acc, prec, rec, f1])
        fold_table = wandb.Table(
            columns=["Fold", "Accuracy", "Precision", "Recall", "F1-Score"],
            data=fold_table_data
        )
        # Use a step higher than max training step for summary logs
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log({"overall/per_fold_summary": fold_table}, step=summary_step)
    
    # Identify best fold based on F1 score
    best_fold_idx = np.argmax(fold_metrics_array[:, 3])  # F1 is at index 3
    best_fold_num = fold_results[best_fold_idx][0]
    best_f1 = fold_metrics_array[best_fold_idx, 3]

        # Log aggregated metrics to wandb
    if wandb_initialized:
        # Use a step higher than max training step for summary logs
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log({
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
        }, step=summary_step)
    
    print(f"{'-'*60}")
    print(f"{'Mean':<10} {mean_acc:>12.3f} {mean_prec:>12.3f} {mean_rec:>12.3f} {mean_f1:>12.3f}")
    print(f"{'Std':<10} {std_acc:>12.3f} {std_prec:>12.3f} {std_rec:>12.3f} {std_f1:>12.3f}")
    print(f"{'='*60}")
    
    # Create aggregated confusion matrix
    labels = label_list
    cm_agg = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    
    # Calculate marginals
    row_sums = np.sum(cm_agg, axis=1)
    col_sums = np.sum(cm_agg, axis=0)
    total_sum = np.sum(cm_agg)
    
    # Create display labels with 'Total'
    display_labels = labels + ['Total']
    n_classes = len(labels)
    
    # Plot aggregated confusion matrix with separate handling for margins
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot the core confusion matrix (without margins) with colormap
    im = ax.imshow(cm_agg, interpolation='nearest', cmap='Blues')
    
    # Add text annotations for the core matrix
    thresh = cm_agg.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm_agg[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm_agg[i, j] > thresh else "black",
                   fontsize=12)
    
    # Extend axes to accommodate margins
    ax.set_xlim(-0.5, n_classes + 0.5)
    ax.set_ylim(n_classes + 0.5, -0.5)
    
    # Add gray background rectangles for margin cells
    for i in range(n_classes):
        # Right column (row sums)
        rect = plt.Rectangle((n_classes - 0.5, i - 0.5), 1, 1, 
                             facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(n_classes, i, format(row_sums[i], 'd'),
               ha="center", va="center", color="black", fontsize=12)
        
        # Bottom row (column sums)
        rect = plt.Rectangle((i - 0.5, n_classes - 0.5), 1, 1,
                             facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(i, n_classes, format(col_sums[i], 'd'),
               ha="center", va="center", color="black", fontsize=12)
    
    # Bottom-right corner (total)
    rect = plt.Rectangle((n_classes - 0.5, n_classes - 0.5), 1, 1,
                         facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(n_classes, n_classes, format(total_sum, 'd'),
           ha="center", va="center", color="black", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_classes + 1))
    ax.set_yticks(np.arange(n_classes + 1))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    
    # Calculate per-class metrics for display
    per_class_precision = precision_score(all_y_true, all_y_pred, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(all_y_true, all_y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(all_y_true, all_y_pred, average=None, labels=labels, zero_division=0)
    
    # Log per-class metrics to wandb
    if wandb_initialized:
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[f"overall/class_{label}_precision"] = per_class_precision[i]
            per_class_metrics[f"overall/class_{label}_recall"] = per_class_recall[i]
            per_class_metrics[f"overall/class_{label}_f1"] = per_class_f1[i]
        # Use a step higher than max training step for summary logs
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log(per_class_metrics, step=summary_step)
    
    # Build detailed title
    title_parts = [loss_name]
    if use_class_weights:
        title_parts.append("Weighted")
    if use_focal_loss and not use_ordinal_loss:
        title_parts.append("Focal")
    loss_title = " + ".join(title_parts) if len(title_parts) > 1 else title_parts[0]
    
    metrics_str = (
        f"Accuracy: {overall_acc:.3f} | "
        f"Precision: {overall_prec:.3f} | "
        f"Recall: {overall_rec:.3f} | "
        f"F1: {overall_f1:.3f}"
    )
    
    ax.set_title(f"AGGREGATED Confusion Matrix ({n_splits}-Fold CV) - {loss_title}\n"
                 f"Total Samples: {len(all_y_true)}\n{metrics_str}", fontsize=12)
                 
    
    # Add per-class metrics as text below the matrix
    class_metrics_text = "Per-Class Metrics:\n"
    for i, label in enumerate(labels):
        class_metrics_text += f"  Class {label}: Prec={per_class_precision[i]:.3f}, Rec={per_class_recall[i]:.3f}, F1={per_class_f1[i]:.3f}\n"
    
    plt.figtext(0.5, 0.02, class_metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    agg_cm_path = VIZUALIZE_DIR + f"aggregated_confusion_matrix.png"
    plt.savefig(agg_cm_path, bbox_inches='tight', dpi=150)
    
    # Log aggregated confusion matrix to wandb
    if wandb_initialized:
        # Use a step higher than max training step for summary logs
        summary_step = max_training_step + 1 if max_training_step > 0 else max_epochs + 1
        wandb.log({"overall/aggregated_confusion_matrix": wandb.Image(fig)}, step=summary_step)
    
    plt.show()
    print(f"\nSaved aggregated confusion matrix to: {agg_cm_path}")
    
    # Print classification report
    print(f"\nClassification Report (Aggregated):")
    print(classification_report(all_y_true, all_y_pred, labels=labels, zero_division=0))
    
    # ============================================================
    # SAVED MODELS SUMMARY
    # ============================================================
    model_save_dir = os.path.join(VIZUALIZE_DIR, 'saved_models')
    print(f"\n{'='*60}")
    print(f"SAVED MODELS SUMMARY")
    print(f"{'='*60}")
    print(f"Models saved to: {model_save_dir}")
    print(f"\nAvailable models:")
    for fold_num, val_subs in fold_to_val_subjects.items():
        model_path = os.path.join(model_save_dir, f'model_fold{fold_num}.pth')
        if os.path.exists(model_path):
            print(f"  • model_fold{fold_num}.pth (validation subjects: {', '.join(val_subs)})")
    
    print(f"\n🏆 Best model: Fold {best_fold_num} (F1 = {best_f1:.3f})")
    best_model_path = os.path.join(model_save_dir, f'model_fold{best_fold_num}.pth')
    print(f"   Path: {best_model_path}")
    
    print(f"\n{'─'*60}")
    print(f"TO LOAD A MODEL FOR VALIDATION ON NEW SUBJECTS:")
    print(f"{'─'*60}")
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
    print(f"Cross-validation complete!")
    print(f"{'='*60}\n")
    
    # Finish wandb run
    if wandb_initialized:
        wandb.finish()
        print(f"🔮 Wandb run completed and synced")

if __name__ == '__main__':
    if validation_only_mode:
        validate_model()
    else:
        main()
