from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

def auc_from_probs(probs, y_true, use_ordinal_loss, num_label_classes):
    """
    Compute macro-averaged multi-class AUC from class or ordinal probabilities.
    probs: (N, C) array; C = num_label_classes-1 for ordinal, num_label_classes for CE.
    """
    probs = probs if isinstance(probs, np.ndarray) else np.asarray(probs)
    if use_ordinal_loss:
        s = probs  # (N, K), K = num_label_classes - 1
        K = s.shape[1]
        s_prev = np.concatenate([np.ones((len(s), 1)), s], axis=1)
        s_next = np.concatenate([s, np.zeros((len(s), 1))], axis=1)
        class_probs = s_prev - s_next  # (N, num_label_classes)
    else:
        class_probs = probs
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(
            y_true, class_probs, multi_class='ovr', average='macro',
            labels=np.arange(num_label_classes)
        ))
    except Exception:
        return np.nan


def compute_auc_from_outputs(output_list, y_list, mask_list, use_ordinal_loss, num_label_classes, device):
    """
    Compute macro-averaged multi-class AUC from model outputs.
    output_list: list of tensors (B, C, T), y_list and mask_list same length.
    For ordinal: C = num_label_classes - 1; convert sigmoid outputs to class probs.
    For CE: C = num_label_classes; use softmax probs.
    Returns float AUC or np.nan if undefined (e.g. one class only).
    """
    all_probs = []
    all_y = []
    for output, y_batch, mask in zip(output_list, y_list, mask_list):
        if output is None or mask.sum() == 0:
            continue
        out = output.detach().to(device)
        y_batch = y_batch.to(device)
        mask = mask.to(device)
        B, C, T = out.shape
        if use_ordinal_loss:
            # out: (B, K, T), K = num_label_classes - 1. sigmoid -> P(Y >= k+1)
            s = torch.sigmoid(out)  # (B, K, T)
            # Class probs: P(Y=0)=1-s0, P(Y=k)=s_{k-1}-s_k, P(Y=K)=s_{K-1}
            s = s.permute(0, 2, 1)  # (B, T, K)
            ones = torch.ones((B, T, 1), device=out.device)
            s_prev = torch.cat([ones, s], dim=2)  # (B, T, K+1)
            s_next = torch.cat([s, torch.zeros((B, T, 1), device=out.device)], dim=2)
            class_probs = (s_prev - s_next)  # (B, T, num_label_classes)
        else:
            class_probs = F.softmax(out, dim=1).permute(0, 2, 1)  # (B, T, C)
        # Flatten: (B*T, num_label_classes), (B*T,), (B*T,)
        class_probs = class_probs.reshape(-1, num_label_classes)
        y_flat = y_batch.reshape(-1).long()
        mask_flat = mask.reshape(-1).bool()
        valid = mask_flat
        all_probs.append(class_probs[valid].cpu().numpy())
        all_y.append(y_flat[valid].cpu().numpy())
    if not all_probs:
        return np.nan
    y_true = np.concatenate(all_y, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        auc = roc_auc_score(
            y_true, probs, multi_class='ovr', average='macro',
            labels=np.arange(num_label_classes)
        )
    except Exception:
        return np.nan
    return float(auc)