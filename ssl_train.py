"""
SSL Training Script - Stage 1
Train HARNet10 using self-supervised learning on your chorea detection data.

This script implements the same SSL approach used in the original ssl-wearables:
- Time Reversal (flip)
- Scaling
- Permutation
- Time Warping

The trained encoder will be used in Stage 2 for J-net segmentation.
"""

import preprocessing
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sslmodel import get_sslnet
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
import sys

# Add the ssl-wearables path to import their modules
sys.path.append('/home/netabiran/hd-chorea-detection/torch_hub_cache/OxWearables_ssl-wearables_v1.0.0')
from sslearning.data import data_transformation as transforms

# Constants
RAW_DATA_AND_LABELS_DIR = '/home/netabiran/data_ready/hd_dataset/lab_geneactive/synced_labeled_data'
preprocessing_mode = False  # Set to True first time to process data
curr_dir = os.getcwd()
PROCESSED_DATA_DIR = os.path.join(curr_dir, 'data_ready')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(curr_dir, 'ssl_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SRC_SAMPLE_RATE = int(100)
STD_THRESH = 0.1
WINDOW_SIZE = int(30*10)  # 300 samples at 30Hz = 10 seconds
WINDOW_OVERLAP = int(30*5)

# SSL Training Config
SSL_EPOCHS = 50
SSL_BATCH_SIZE = 64
SSL_LR = 1e-4
SSL_PATIENCE = 15


class SSLTransformDataset(Dataset):
    """
    Dataset that applies SSL transformations to accelerometer windows.
    Each sample returns: (original_data, transformed_data, transformation_labels)
    """
    def __init__(self, X, apply_transforms=True):
        """
        Args:
            X: numpy array of shape [N, 3, 300] (N windows, 3 axes, 300 timesteps)
            apply_transforms: whether to apply SSL transformations
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.apply_transforms = apply_transforms
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        window = self.X[idx].numpy()  # Shape: [3, 300]
        
        if not self.apply_transforms:
            # For validation, return original data with no transforms applied
            labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)
            return torch.tensor(window, dtype=torch.float32), labels
        
        # Apply SSL transformations
        labels = []
        
        # 1. Time Reversal (flip)
        flip_choice = np.random.choice([0, 1], p=[0.5, 0.5])
        window = transforms.flip(window, flip_choice)
        labels.append(flip_choice)
        
        # 2. Scaling
        scale_choice = np.random.choice([0, 1], p=[0.5, 0.5])
        window = transforms.scale(window, scale_choice, scale_range=0.5, min_scale_diff=0.15)
        labels.append(scale_choice)
        
        # 3. Permutation
        permute_choice = np.random.choice([0, 1], p=[0.5, 0.5])
        window = transforms.permute(window, permute_choice, nPerm=4, minSegLength=10)
        labels.append(permute_choice)
        
        # 4. Time Warping
        time_warp_choice = np.random.choice([0, 1], p=[0.5, 0.5])
        window = transforms.time_warp(window, time_warp_choice, sigma=0.2)
        labels.append(time_warp_choice)
        
        return torch.tensor(window, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class SSLModel(nn.Module):
    """
    SSL Model wrapper that adds multi-task heads for the 4 SSL tasks.
    """
    def __init__(self, base_model):
        super(SSLModel, self).__init__()
        self.feature_extractor = base_model.feature_extractor
        
        # Get feature dimension from the base model
        # After feature_extractor, we have [batch, 1024, ~2] depending on pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Binary classification heads for each SSL task
        self.time_reversal_head = nn.Linear(1024, 2)
        self.scale_head = nn.Linear(1024, 2)
        self.permutation_head = nn.Linear(1024, 2)
        self.time_warp_head = nn.Linear(1024, 2)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        # Global pooling
        pooled = self.global_pool(features).squeeze(-1)  # [batch, 1024]
        
        # Task predictions
        time_reversal = self.time_reversal_head(pooled)
        scale = self.scale_head(pooled)
        permutation = self.permutation_head(pooled)
        time_warp = self.time_warp_head(pooled)
        
        return time_reversal, scale, permutation, time_warp


def compute_ssl_loss(preds, labels):
    """
    Compute multi-task SSL loss.
    
    Args:
        preds: tuple of (time_reversal, scale, permutation, time_warp) predictions
        labels: tensor of shape [batch, 4] with binary labels for each task
    
    Returns:
        total_loss, individual_losses, accuracies
    """
    criterion = nn.CrossEntropyLoss()
    
    time_reversal_pred, scale_pred, permutation_pred, time_warp_pred = preds
    
    # Compute losses
    loss_time_reversal = criterion(time_reversal_pred, labels[:, 0])
    loss_scale = criterion(scale_pred, labels[:, 1])
    loss_permutation = criterion(permutation_pred, labels[:, 2])
    loss_time_warp = criterion(time_warp_pred, labels[:, 3])
    
    total_loss = loss_time_reversal + loss_scale + loss_permutation + loss_time_warp
    
    # Compute accuracies
    acc_time_reversal = (time_reversal_pred.argmax(dim=1) == labels[:, 0]).float().mean()
    acc_scale = (scale_pred.argmax(dim=1) == labels[:, 1]).float().mean()
    acc_permutation = (permutation_pred.argmax(dim=1) == labels[:, 2]).float().mean()
    acc_time_warp = (time_warp_pred.argmax(dim=1) == labels[:, 3]).float().mean()
    
    individual_losses = {
        'time_reversal': loss_time_reversal.item(),
        'scale': loss_scale.item(),
        'permutation': loss_permutation.item(),
        'time_warp': loss_time_warp.item()
    }
    
    accuracies = {
        'time_reversal': acc_time_reversal.item(),
        'scale': acc_scale.item(),
        'permutation': acc_permutation.item(),
        'time_warp': acc_time_warp.item(),
        'average': (acc_time_reversal + acc_scale + acc_permutation + acc_time_warp).item() / 4
    }
    
    return total_loss, individual_losses, accuracies


def train_ssl_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_accs = []
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        preds = model(data)
        loss, individual_losses, accs = compute_ssl_loss(preds, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_accs.append(accs['average'])
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, "
                  f"Avg Acc: {accs['average']:.4f}")
    
    return total_loss / len(dataloader), np.mean(all_accs)


def validate_ssl(model, dataloader, device):
    """Validate the SSL model."""
    model.eval()
    total_loss = 0
    all_accs = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            loss, individual_losses, accs = compute_ssl_loss(preds, labels)
            
            total_loss += loss.item()
            all_accs.append(accs['average'])
    
    return total_loss / len(dataloader), np.mean(all_accs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or preprocess data
    if preprocessing_mode:
        print("=== Preprocessing Data ===")
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

        res = {
            'win_data_all_sub': win_data_all_sub,
            'win_labels_all_sub': win_labels_all_sub,
            'win_subjects': win_subjects,
            'win_chorea_all_sub': win_chorea_all_sub,
            'win_shift_all_sub': win_shift_all_sub,
            'win_video_time_all_sub': win_video_time_all_sub
        }
        np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model.npz'), **res)
    
    # Load processed data
    print("\n=== Loading Processed Data ===")
    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model.npz'))
    
    win_acc_data = input_file['win_data_all_sub']
    win_acc_data = np.transpose(win_acc_data, [0, 2, 1])  # [N, 300, 3]
    win_subjects = input_file['win_subjects']
    
    print(f"Total windows: {win_acc_data.shape[0]}")
    print(f"Window shape: {win_acc_data.shape[1:]}")
    
    # Transpose to [N, 3, 300] for model input
    win_acc_data = np.transpose(win_acc_data, [0, 2, 1])
    
    # Setup GroupKFold cross-validation
    subjects = np.array([str(s) for s in win_subjects.reshape(-1)])
    unique_subjects = np.unique(subjects)
    unique_subjects.sort()
    subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
    groups = np.array([subject_to_idx[s] for s in subjects])
    
    n_splits = min(5, len(unique_subjects))
    gkf = GroupKFold(n_splits=n_splits)
    
    print(f"\n=== Starting SSL Training with {n_splits}-Fold CV ===")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(win_acc_data, groups=groups, groups=groups)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        X_train, X_val = win_acc_data[train_idx], win_acc_data[val_idx]
        
        print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
        
        # Create datasets
        train_dataset = SSLTransformDataset(X_train, apply_transforms=True)
        val_dataset = SSLTransformDataset(X_val, apply_transforms=True)
        
        train_loader = DataLoader(train_dataset, batch_size=SSL_BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=SSL_BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Initialize model
        print("\n=== Initializing SSL Model ===")
        base_model = get_sslnet(tag='v1.0.0', pretrained=True, num_classes=2, model_type='vanila')
        ssl_model = SSLModel(base_model).to(device)
        
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=SSL_LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                factor=0.5, patience=5, verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(SSL_EPOCHS):
            print(f"\nEpoch {epoch+1}/{SSL_EPOCHS}")
            print("-" * 40)
            
            train_loss, train_acc = train_ssl_epoch(ssl_model, train_loader, optimizer, device)
            val_loss, val_acc = validate_ssl(ssl_model, val_loader, device)
            
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': ssl_model.state_dict(),
                    'feature_extractor_state_dict': ssl_model.feature_extractor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, os.path.join(OUTPUT_DIR, f'ssl_model_fold{fold}_best.pth'))
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= SSL_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print(f"\nFold {fold+1} completed. Best val loss: {best_val_loss:.4f}")
    
    print("\n" + "="*60)
    print("SSL Training Complete!")
    print(f"Models saved in: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()

