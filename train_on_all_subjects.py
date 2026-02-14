from model_evaluation import compute_auc_from_outputs
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import numpy as np
import os


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
        
        # Check for improvement (using training loss since no validation set)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improvement_marker = " ✓ (best)"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""
        
        print(f"Epoch {epoch + 1:3d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc_str} | LR: {learning_rate:.2e}{improvement_marker}")
        
        # Log to wandb
        if wandb_initialized:
            log_dict = {
                "train_loss": avg_train_loss,
                "learning_rate": learning_rate,
                "epoch": epoch + 1,
            }
            if not np.isnan(train_auc):
                log_dict["train_auc"] = train_auc
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
