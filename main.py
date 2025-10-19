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

curr_dir = os.getcwd()

PROCESSED_DATA_DIR = os.path.join(curr_dir, 'data_ready')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(curr_dir, 'model_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    left_wind = left[final_mask]
    mid_wind = mid[final_mask]
    right_wind = right[final_mask]
    mid_chorea = mid_chorea[final_mask]
    win_subjects = win_subjects[1:-1:2][final_mask]

    win_chorea = mid_chorea
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
    feature_list = [extract_features(win_acc_data[i]) for i in range(win_acc_data.shape[0])]
    window_features = np.stack(feature_list)  # shape: [num_windows, num_features]
    print("Window features shape:", window_features.shape)


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


    # class MaskedCrossEntropyLoss(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()

    #     def forward(self, logits, labels, mask):
    #         # Compute un-reduced loss per element
    #         loss = F.cross_entropy(logits, labels, reduction='none')  # shape: [batch, time] or [N]
            
    #         # Apply the mask
    #         masked_loss = loss * mask

    #         # Normalize by number of valid elements
    #         num_valid = mask.sum()
    #         return masked_loss.sum() / (num_valid + 1e-5)

    class CumulativeOrdinalLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, logits, labels, mask):
            B, K, T = logits.shape
            labels_bin = torch.zeros((B, K, T), device=logits.device)
            for k in range(K):
                labels_bin[:, k, :] = (labels >= (k + 1)).float()
            loss = self.bce(logits, labels_bin)
            loss = loss * mask.unsqueeze(1)
            return loss.sum() / (mask.sum() * K + 1e-8)
        
    # ------------------ Axis normalization ------------------
    def normalize_axes(X):
        """
        X: [num_samples, 3, num_timesteps]
        Returns X_rot: rotated axes where:
        - first axis = main movement (PCA)
        - second axis = orthogonal
        - third axis = gravity aligned
        """
        X_rot = np.zeros_like(X)
        for i in range(X.shape[0]):
            window = X[i].T  # [timesteps, 3]

            # 1. Gravity vector
            g = window.mean(axis=0)
            g = g / (np.linalg.norm(g) + 1e-8)

            # 2. Remove gravity
            window_no_g = window - g * np.dot(window, g[:, None])

            # 3. PCA on remaining movement
            cov = np.cov(window_no_g, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            R = eigvecs[:, order]

            # 4. Rotate window
            X_rot[i] = (window @ R).T
        return X_rot

        # ------------------------------
    # GroupKFold subject-wise CV
    # ------------------------------

    # --- Axis normalization
    win_acc_data = normalize_axes(win_acc_data)

    subjects = np.array([str(s) for s in win_subjects.reshape(-1)])
    n_splits = min(5, len(np.unique(subjects)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(win_acc_data, win_chorea, groups=subjects)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        X_train, y_train = win_acc_data[train_idx], win_chorea[train_idx]
        X_val, y_val = win_acc_data[val_idx], win_chorea[val_idx]

        mask_train = (y_train >= 0).astype(float)
        mask_val = (y_val >= 0).astype(float)

        y_train = np.maximum(y_train, 0)
        y_val = np.maximum(y_val, 0)

        # Select features corresponding to this fold
        feats_train = window_features[train_idx]
        feats_val = window_features[val_idx]

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
        model = get_sslnet(tag='v1.0.0', pretrained=True,
                           num_classes=5, model_type='segmentation',
                           padding_type='triple_wind')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = CumulativeOrdinalLoss()

        # === Training ===
        model.train()
        for epoch in range(20):
            total_loss = 0
            for X_batch, y_batch, mask, feats in train_loader:
                X_batch, y_batch, mask, feats = X_batch.to(device), y_batch.to(device).long(), mask.to(device), feats.to(device)    
                optimizer.zero_grad()
                output = model(X_batch, feats)
                loss = criterion(output, y_batch, mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        # === Evaluation on validation fold ===
        model.eval()
        all_preds, all_labels, all_masks = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, mask, feats in val_loader:
                X_batch = X_batch.to(device)
                feats = feats.to(device)
                logits = model(X_batch, feats).cpu()
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).sum(dim=1)
                all_preds.append(pred)
                all_labels.append(y_batch)
                all_masks.append(mask)

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

    # === Average results across folds ===
    avg_results = np.mean(fold_results, axis=0)
    print(f"\nAverage across folds → Acc: {avg_results[0]:.3f}, Prec: {avg_results[1]:.3f}, "
          f"Rec: {avg_results[2]:.3f}, F1: {avg_results[3]:.3f}")

    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Add marginals (row and column sums)
    cm_with_margins = np.zeros((cm.shape[0]+1, cm.shape[1]+1), dtype=int)
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
    title_str = (
        f"Accuracy: {acc:.3f} | "
        f"Precision: {prec:.3f} | "
        f"Recall: {rec:.3f} | "
        f"F1 Score: {f1:.3f}"
    )
    ax.set_title(f"Confusion Matrix (Chorea 0–4)\n{title_str}", fontsize=12)

    plt.tight_layout()
    plt.savefig("/home/netabiran/hd-chorea-detection/figures_output/multiclass_new_axes/conf_matrix_new_feature_engineering.png")
    plt.show()

if __name__ == '__main__':
    main()
