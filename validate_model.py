import numpy as np
import os
import torch
from sslmodel import get_sslnet
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from model_evaluation import auc_from_probs

def validate_model(validation_model_path, validation_subjects, window_size=30, PROCESSED_DATA_DIR='processed_data', VIZUALIZE_DIR='vizualize'):
    """
    Validation-only mode: Load a saved model and evaluate on specific subjects.
    Uses the same preprocessing pipeline as training.
    """
    
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
    win_acc_data = np.stack([left_wind, mid_wind, right_wind], axis=2).reshape(-1, 3, window_size * 3)
    
    # Extract features if needed
    if model_use_features:
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
    # AUC from collected probs (flatten and mask)
    probs_cat = torch.cat(all_probs, dim=0).permute(0, 2, 1).reshape(-1, all_probs[0].shape[1])
    probs_valid = probs_cat[valid_mask].numpy()
    auc = auc_from_probs(probs_valid, y_true, model_use_ordinal_loss, num_label_classes)
    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
    
    print(f"\nOverall Metrics:")
    print(f"  • Total valid samples: {len(y_true)}")
    print(f"  • Accuracy:  {acc:.3f}")
    print(f"  • Precision: {prec:.3f}")
    print(f"  • Recall:    {rec:.3f}")
    print(f"  • F1-Score:  {f1:.3f}")
    print(f"  • AUC:       {auc_str}")
    
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
    metrics_str = f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc_str}"
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