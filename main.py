import preprocessing
import numpy as np
import os
import torch
from sslmodel import get_sslnet
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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
use_features = True
use_ordinal_loss = True  # True: ordinal loss (2 outputs for 3 classes), False: masked cross-entropy (3 outputs)
use_ssl_encoder = False  # True: load SSL-trained encoder if available, False: use original pretrained
use_class_weights = True  # True: use class weights to handle imbalanced data, False: no weighting
use_focal_loss = False  # True: use focal loss (only for masked CE), False: standard loss

curr_dir = os.getcwd()

PROCESSED_DATA_DIR = os.path.join(curr_dir, 'data_ready')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(curr_dir, 'model_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SSL_OUTPUT_DIR = os.path.join(curr_dir, 'ssl_outputs')

VIZUALIZE_DIR = os.path.join(curr_dir, 'model_outputs', 'results_visualization')
os.makedirs(VIZUALIZE_DIR, exist_ok=True)

SRC_SAMPLE_RATE = int(100)
STD_THRESH = 0.1
WINDOW_SIZE = int(30*10)
WINDOW_OVERLAP = int(30*5)  

def main():
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

        np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model.npz'), **res)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    print("start loading input file")
    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model.npz'))
    # input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_hd_only_segmentation_triple_wind_no_shift.npz'))
    print("done loading input file")

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

    win_chorea = mid_chorea
    
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
    n_splits = min(5, len(unique_subjects))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    
    # Determine number of classes and loss type based on configuration
    if use_ordinal_loss:
        num_classes = 2  # Ordinal loss: 2 outputs for 3 levels (0, 1, 2)
        loss_name = "Ordinal"
        print(f"\n{'='*60}")
        print(f"Using Cumulative Ordinal Loss with {num_classes} outputs for 3 classes")
        print(f"{'='*60}")
    else:
        num_classes = 3  # Masked CE: 3 outputs (one per class)
        loss_name = "MaskedCE"
        print(f"\n{'='*60}")
        print(f"Using Masked Cross-Entropy Loss with {num_classes} outputs for 3 classes")
        print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(win_acc_data, win_chorea, groups=groups)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        X_train, y_train = win_acc_data[train_idx], win_chorea[train_idx]
        X_val, y_val = win_acc_data[val_idx], win_chorea[val_idx]

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
            # Initialize weights array for all possible classes (0, 1, 2)
            num_total_classes = 3
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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
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

        # === Training ===
        model.train()
        for epoch in range(20):
            total_loss = 0
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
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

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

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Fold {fold+1} → Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        fold_results.append((acc, prec, rec, f1))

        # === Confusion Matrix ===
        labels = [0, 1, 2]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Add marginals (row and column sums)
        cm_with_margins = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1), dtype=int)
        cm_with_margins[:-1, :-1] = cm
        cm_with_margins[:-1, -1] = np.sum(cm, axis=1)
        cm_with_margins[-1, :-1] = np.sum(cm, axis=0)
        cm_with_margins[-1, -1] = np.sum(cm)

        # Create new display labels with 'Total'
        display_labels = labels + ['Total']

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_with_margins, display_labels=display_labels)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)

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
        plt.savefig(f"/home/netabiran/hd-chorea-detection/figures_output/segmentation_combined_labels/conf_matrix_no_axis_rotation_with_features_combined_labels_with_training_3_classes_ordinal_weighted_new_trial.png")
        plt.show()

if __name__ == '__main__':
    main()
