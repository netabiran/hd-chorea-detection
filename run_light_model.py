import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ---- your preprocessing module (uses bandpass_filter and data_windowing) ----
import preprocessing

# ===================== USER CONFIG (Windows paths) ===================== #
RAW_DATA_AND_LABELS_DIR = r'/home/netabiran/data_ready/hd_dataset/lab_geneactive/synced_labeled_data' 
WORK_DIR = os.getcwd()

PROCESSED_DATA_DIR = os.path.join(WORK_DIR, 'data_ready_basic_cnn')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(WORK_DIR, 'model_outputs_basic_cnn')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIZ_DIR = os.path.join(OUTPUT_DIR, 'results_visualization')
os.makedirs(VIZ_DIR, exist_ok=True)

# ===================== PARAMETERS ===================== #
preprocessing_mode = True  # set True once to generate the NPZ, then False to skip
SRC_SAMPLE_RATE = 100            # original sampling rate of acc_data
LOW_CUT, HIGH_CUT, ORDER = 0.2, 15, 4
WINDOW_SIZE = 25 * 1         
WINDOW_OVERLAP = 25 * 0.5          
STD_THRESH = 0.1
N_CLASSES = 5                    # chorea 0..4
NPZ_NAME = 'windows_input_to_multiclass_model_basic_0.5_sec.npz'  # saved the same way as before

# ===================== HELPERS ===================== #

def majority_label_per_window(labels_window):
    """
    labels_window: shape (T,) or (T,1). Values in [0..4] or -1 for invalid.
    Return an integer in [0..4] or None if no valid labels in window.
    """
    lbl = np.array(labels_window).reshape(-1)
    valid = lbl[lbl >= 0]
    if valid.size == 0:
        return None
    # majority vote (or median – majority tends to be intuitive here)
    vals, counts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(counts)])

def valid_window(labels_window, chorea_window):
        labels_window = np.array(labels_window).reshape(-1)
        chorea_window = np.array(chorea_window).reshape(-1)

        walking_ok = np.sum(labels_window > 0) > 0
        chorea_ok = np.sum(chorea_window >= 0) >= 2
        return walking_ok and chorea_ok

def make_1s_windows(acc, labels, chorea, video_time,
                    window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    """
    Generates windows from acc data, applies gait + chorea filter.
    Only returns windows passing valid_window().
    """
    step = int(window_size - overlap)
    window_size = int(window_size)
    n = acc.shape[0]

    windows = []
    labels_list = []
    chorea_list = []
    times_list = []
    shift_list = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        lbl_win = labels[start:end]
        chorea_win = chorea[start:end] 

        # Apply your gait + chorea filter
        if not valid_window(lbl_win, chorea_win):
            continue

        windows.append(acc[start:end].T)  # (3, window_size)
        labels_list.append(lbl_win)
        chorea_list.append(chorea_win)
        shift_list.append(np.zeros_like(lbl_win))
        if video_time is not None:
            times_list.append([video_time[start]])
        else:
            times_list.append([0])

    if len(windows) == 0:
        return None, None, None, None, None

    data_w = np.array(windows)
    labels_w = np.array(labels_list)
    chorea_w = np.array(chorea_list)
    video_time_w = np.array(times_list)
    shift_w = np.array(shift_list)

    return data_w, labels_w, chorea_w, video_time_w, shift_w


class WindowDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, C, T)
        # y: (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BasicCNN(nn.Module):
    """
    Simple/shallow 1D CNN over (C=3, T=WINDOW_SIZE) windows.
    """
    def __init__(self, n_classes=5, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (B, 64)
        x = self.drop(x)
        return self.fc(x)             # (B, n_classes)
    

# ===================== PREPROCESS (no resample) ===================== #

def preprocess_and_save():
    """
    - Loads each npz under RAW_DATA_AND_LABELS_DIR
    - Band-pass filter ONLY (no resampling)
    - Windowing via your preprocessing.data_windowing
    - Saves NPZ with same fields you used before (so downstream is consistent)
    """
    win_data_all_sub = np.empty((0, 3, WINDOW_SIZE))  # (N, C, T)
    win_labels_all_sub = np.empty((0, WINDOW_SIZE))   # per-frame labels in window
    win_subjects = np.empty((0, 1), dtype=object)
    win_chorea_all_sub = np.empty((0, WINDOW_SIZE))   # if present / needed
    win_shift_all_sub = np.empty((0, WINDOW_SIZE))
    win_video_time_all_sub = np.empty((0, 1))

    for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
        path = os.path.join(RAW_DATA_AND_LABELS_DIR, file)
        if not file.lower().endswith('.npz'):
            continue
        try:
            data_file = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Can't open {file}: {e}")
            continue

        # acc_data may be saved as numeric or strings; make numeric
        try:
            acc = data_file['acc_data'].astype(float)  # (T, 3) or (N, 3)
        except Exception:
            # robust numeric conversion
            acc0 = data_file['acc_data']
            def is_numeric(v):
                try:
                    float(v)
                    return True
                except:
                    return False
            mask = np.vectorize(is_numeric)(acc0)
            acc0 = acc0.astype(object)
            acc0[~mask] = np.nan
            acc = acc0.astype(float)
            acc = acc[~np.isnan(acc).any(axis=1)]
            if acc.size == 0:
                print(f"Skipping {file}, no valid acc_data after cleaning")
                continue

        labels = data_file.get('label_data', None)        # (T,)
        chorea = data_file.get('chorea_labels', None)     # (T,) or None
        video_time = data_file.get('time_data', None)     # (T,) or None
        subject_name = file.split('.')[0]

        # Band-pass ONLY (no resample)
        acc = preprocessing.bandpass_filter(
            data=acc, low_cut=LOW_CUT, high_cut=HIGH_CUT,
            sampling_rate=SRC_SAMPLE_RATE, order=ORDER
        )

        # Windowing (no resample). This call should mirror your existing behavior.
        data, labels_w, chorea_w, video_time_w, shift_w = make_1s_windows(
            acc, labels, chorea, video_time,
            window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)

        # data: (Nw, 3, WINDOW_SIZE)
        # labels_w: (Nw, WINDOW_SIZE)   values in [0..4] or -1 for invalid
        # chorea_w, video_time_w, shift_w shape aligned

        if data is None or len(data) == 0:
            continue

        # append
        win_data_all_sub = np.append(win_data_all_sub, data, axis=0)
        win_labels_all_sub = np.append(win_labels_all_sub, labels_w, axis=0)
        if chorea_w is None:
            chorea_w = -1 * np.ones_like(labels_w)
        win_chorea_all_sub = np.append(win_chorea_all_sub, chorea_w, axis=0)
        win_shift_all_sub = np.append(win_shift_all_sub, shift_w, axis=0)

        if video_time_w is None:
            video_time_w = np.zeros((data.shape[0], 1))
        win_video_time_all_sub = np.append(win_video_time_all_sub, video_time_w, axis=0)

        subjects_col = np.tile(subject_name, (data.shape[0], 1))
        win_subjects = np.append(win_subjects, subjects_col, axis=0)

        print(f"{file} -> windows: {data.shape[0]} | total: {win_data_all_sub.shape[0]}")

    # Save like before
    res = {
        'win_data_all_sub': win_data_all_sub,
        'win_labels_all_sub': win_labels_all_sub,
        'win_subjects': win_subjects,
        'win_chorea_all_sub': win_chorea_all_sub,
        'win_shift_all_sub': win_shift_all_sub,
        'win_video_time_all_sub': win_video_time_all_sub
    }
    np.savez(os.path.join(PROCESSED_DATA_DIR, NPZ_NAME), **res)
    print(f"Saved {os.path.join(PROCESSED_DATA_DIR, NPZ_NAME)}")

# ===================== LOSS FUNCTION ===================== #
# ----- Weighted Cumulative Ordinal Loss -----
class CumulativeOrdinalLossWithWeights(torch.nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.class_weights = class_weights  # tensor of shape (K,)

    def forward(self, logits, labels, mask):
        B, K, T = logits.shape
        labels_bin = torch.zeros((B, K, T), device=logits.device)
        for k in range(K):
            labels_bin[:, k, :] = (labels >= (k + 1)).float()
        loss = self.bce(logits, labels_bin)  # (B, K, T)
        # apply class weights along K
        loss = loss * self.class_weights[:K].view(1, K, 1)
        loss = loss * mask.unsqueeze(1)
        return loss.sum() / (mask.sum() * K + 1e-8)

# ===================== TRAIN / EVAL ===================== #

def train_and_eval():
    print("Loading:", os.path.join(PROCESSED_DATA_DIR, NPZ_NAME))
    data = np.load(os.path.join(PROCESSED_DATA_DIR, NPZ_NAME), allow_pickle=True)

    X = data['win_data_all_sub']           # (N, 3, T)
    chorea_win = data['win_chorea_all_sub']     # (N, T) values 0..4 or -1
    # Convert each window to a single class label (0..4) by majority vote
    y = []
    keep_idx = []
    for i in range(chorea_win.shape[0]):
        lbl = majority_label_per_window(chorea_win[i])
        if lbl is not None:
            y.append(lbl)
            keep_idx.append(i)

    X = X[keep_idx]
    y = np.array(y, dtype=int)

    # ----------------- Train / Eval ----------------- #
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Compute class weights (inverse frequency)
    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-8)  # avoid division by zero
    class_weights = class_weights / class_weights.sum() * N_CLASSES  # normalize

    # DataLoaders
    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(WindowDataset(X_test,  y_test), batch_size=64, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model (output now logits for K = N_CLASSES-1 thresholds)
    K = N_CLASSES - 1
    model = BasicCNN(n_classes=K, in_channels=3).to(device)
    
    # Weighted ordinal loss
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = CumulativeOrdinalLossWithWeights(class_weights_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # --- Training ---
    EPOCHS = 20
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)                # (B, K)
            logits = logits.unsqueeze(-1)     # (B, K, T=1)
            mask = torch.ones_like(yb, dtype=torch.float32, device=device).unsqueeze(-1)
            loss = criterion(logits, yb.unsqueeze(-1), mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_loader):.4f}")

        # --- Eval ---
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)               # (B, K)
                pred = (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().numpy()
                all_preds.append(pred)
                all_trues.append(yb.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_trues)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f"[TEST] Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

        # Confusion Matrix
        labels = list(range(N_CLASSES))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f"BasicCNN (window-level) | Acc={acc:.3f}, F1={f1:.3f}")
        plt.tight_layout()
        out_png = os.path.join(VIZ_DIR, 'conf_matrix_basic_cnn_weighted_ordinal_0.5_sec_window.png')
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Saved CM to:", out_png)

def main():
    if preprocessing_mode:
        preprocess_and_save()
    train_and_eval()

if __name__ == '__main__':
    main()
