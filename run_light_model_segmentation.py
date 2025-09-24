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

# ---- your preprocessing module ----
import preprocessing

# ===================== USER CONFIG ===================== #
RAW_DATA_AND_LABELS_DIR = r'/home/netabiran/data_ready/hd_dataset/lab_geneactive/synced_labeled_data'
WORK_DIR = os.getcwd()

PROCESSED_DATA_DIR = os.path.join(WORK_DIR, 'data_ready_basic_cnn')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(WORK_DIR, 'model_outputs_basic_cnn')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIZ_DIR = os.path.join(OUTPUT_DIR, 'results_visualization')
os.makedirs(VIZ_DIR, exist_ok=True)

# ===================== PARAMETERS ===================== #
use_frequency_domain = True   # <--- TOGGLE HERE
preprocessing_mode = False     # set True once to regenerate NPZ

SRC_SAMPLE_RATE = 100
LOW_CUT, HIGH_CUT, ORDER = 0.2, 15, 4
WINDOW_SIZE = 100 * 1
WINDOW_OVERLAP = 100 * 0.5
STD_THRESH = 0.1
N_CLASSES = 5
NPZ_NAME = 'windows_input_to_multiclass_model_basic_0.5_sec.npz'

# ===================== HELPERS ===================== #
def majority_label_per_window(labels_window):
    lbl = np.array(labels_window).reshape(-1)
    valid = lbl[lbl >= 0]
    if valid.size == 0:
        return None
    vals, counts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(counts)])

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===================== MODELS ===================== #
class BasicCNN_TimeDomain(nn.Module):
    """1D CNN over (C, T), per-frame output (ordinal)."""
    def __init__(self, n_classes=5, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Conv1d(64, n_classes - 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return self.fc(x)   # (B, K, T)

class BasicCNN_FreqDomain(nn.Module):
    """CNN over frequency features, outputs per-window logits."""
    def __init__(self, n_classes=5, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, C, F)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)   # global average pool over frequency bins
        x = self.drop(x)
        return self.fc(x)    # (B, N_CLASSES)

# ===================== LOSS FUNCTIONS ===================== #
class CumulativeOrdinalLossWithWeights(torch.nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.class_weights = class_weights

    def forward(self, logits, labels, mask):
        B, K, T = logits.shape
        labels_bin = torch.zeros((B, K, T), device=logits.device)
        for k in range(K):
            labels_bin[:, k, :] = (labels >= (k + 1)).float()
        loss = self.bce(logits, labels_bin)
        loss = loss * self.class_weights[:K].view(1, K, 1)
        loss = loss * mask.unsqueeze(1)
        return loss.sum() / (mask.sum() * K + 1e-8)

# ===================== TRAIN / EVAL ===================== #
def train_and_eval():
    print("Loading:", os.path.join(PROCESSED_DATA_DIR, NPZ_NAME))
    data = np.load(os.path.join(PROCESSED_DATA_DIR, NPZ_NAME), allow_pickle=True)

    X = data['win_data_all_sub']       # (N, 3, T)
    y = data['win_chorea_all_sub']     # (N, T), values 0..4 or -1

    # Switch label format
    if use_frequency_domain:
        # Convert to frequency domain
        X_fft = np.fft.rfft(X, axis=2)
        X = np.abs(X_fft)              # magnitude spectrum
        # majority label per window
        y = np.array([majority_label_per_window(lbl) for lbl in y])
        keep_idx = [i for i in range(len(y)) if y[i] is not None]
        X, y = X[keep_idx], y[keep_idx]
    else:
        # keep per-frame labels
        keep_idx = [i for i in range(y.shape[0]) if np.any(y[i] >= 0)]
        X, y = X[keep_idx], y[keep_idx]
        y[y < 0] = -100

    # ----------------- Train / Eval ----------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    if use_frequency_domain:
        # class weights
        class_counts = np.bincount(y_train, minlength=N_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * N_CLASSES

        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_loader  = DataLoader(WindowDataset(X_test,  y_test), batch_size=64, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BasicCNN_FreqDomain(n_classes=N_CLASSES, in_channels=3).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # training
        EPOCHS = 20
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[FREQ] Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_loader):.4f}")

        # eval
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_trues)

    else:
        # class weights
        flat_y = y_train[y_train >= 0].astype(int)
        class_counts = np.bincount(flat_y, minlength=N_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * N_CLASSES

        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_loader  = DataLoader(WindowDataset(X_test,  y_test), batch_size=64, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BasicCNN_TimeDomain(n_classes=N_CLASSES, in_channels=3).to(device)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = CumulativeOrdinalLossWithWeights(class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # training
        EPOCHS = 20
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                mask = (yb != -100).float()
                loss = criterion(logits, yb, mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[TIME] Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_loader):.4f}")

        # eval
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = (torch.sigmoid(logits) > 0.5).sum(dim=1)
                all_preds.append(pred.cpu().numpy().reshape(-1))
                all_trues.append(yb.cpu().numpy().reshape(-1))
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_trues)
        mask = y_true >= 0
        y_pred, y_true = y_pred[mask], y_true[mask]

    # ----------------- Metrics + Confusion Matrix ----------------- #
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"[TEST] Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(N_CLASSES)))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    mode_name = "FreqDomain" if use_frequency_domain else "TimeDomain"
    ax.set_title(f"BasicCNN ({mode_name}) | Acc={acc:.3f}, F1={f1:.3f}")
    plt.tight_layout()
    out_png = os.path.join(VIZ_DIR, f'conf_matrix_basic_cnn_{mode_name}_100_frames_window.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved CM to:", out_png)

def main():
    train_and_eval()

if __name__ == '__main__':
    main()
