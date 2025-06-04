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

    print("start loading input file")
    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model.npz'))
    print("done loading input file")

    input_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model.npz'))
    print("done loading input file")

    win_acc_data = input_file['win_data_all_sub']
    win_acc_data = np.transpose(win_acc_data,[0,2,1])
    win_labels = input_file['win_labels_all_sub']
    win_chorea = input_file['win_chorea_all_sub']
    win_subjects = input_file['win_subjects']
    win_shift = input_file['win_shift_all_sub']
    win_shift = np.mean(win_shift, axis=-1)

    # Filter valid middle windows
    left_wind = win_acc_data[0:-2:2]
    mid_wind = win_acc_data[1:-1:2]
    right_wind = win_acc_data[2::2]
    mid_labels = win_labels[1:-1:2]
    mid_chorea = win_chorea[1:-1:2]

    # Only keep if middle window contains any walking & at least 2 valid chorea samples
    walking_mask = (mid_labels > 0).sum(axis=1) > 0
    valid_chorea_mask = (mid_chorea >= 0).sum(axis=1) >= 2
    final_mask = walking_mask & valid_chorea_mask

    left_wind = left_wind[final_mask]
    mid_wind = mid_wind[final_mask]
    right_wind = right_wind[final_mask]
    mid_chorea = mid_chorea[final_mask]

    # Stack windows
    win_acc_data = np.stack([left_wind, mid_wind, right_wind], axis=2).reshape(-1, 3, WINDOW_SIZE * 3)
    win_chorea = mid_chorea

    # === Dataloader preparation ===
    class ChoreaDataset(Dataset):
        def __init__(self, X, y, mask):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.mask = torch.tensor(mask, dtype=torch.float32)
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.mask[idx]

    class MaskedCrossEntropyLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        def forward(self, input, target, mask):
            loss = self.ce(input, target)
            masked_loss = loss * mask
            return masked_loss.sum() / (mask.sum() + 1e-6)

    # Split randomly regardless of subject
    X_train, X_test, y_train, y_test = train_test_split(
        win_acc_data, win_chorea, test_size=0.2, random_state=42, shuffle=True
    )

    valid_train = (y_train >= 0).astype(float)
    valid_test = (y_test >= 0).astype(float)
    y_train = np.maximum(y_train, 0)
    y_test = np.maximum(y_test, 0)

    train_dataset = ChoreaDataset(X_train, y_train, valid_train)
    test_dataset = ChoreaDataset(X_test, y_test, valid_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # === Model setup ===
    model = get_sslnet(tag='v1.0.0', pretrained=True, num_classes=5, model_type='segmentation', padding_type='triple_wind')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MaskedCrossEntropyLoss()

    # === Training ===
    model.train()
    for epoch in range(5):
        total_loss = 0
        for X_batch, y_batch, mask in train_loader:
            X_batch, y_batch, mask = X_batch.to(device), y_batch.to(device).long(), mask.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # === Evaluation ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).cpu()
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs)
            all_labels.append(y_batch)

    all_preds = torch.cat(all_preds, dim=0).permute(0, 2, 1).contiguous()
    all_labels = torch.cat(all_labels, dim=0)
    y_score = all_preds.view(-1, 5)
    y_true = all_labels.view(-1).long()
    valid_mask = y_true >= 0
    y_score = y_score[valid_mask]
    y_true = y_true[valid_mask]

    unique_classes = torch.unique(y_true)
    print("Classes in test set:", unique_classes.tolist())
    if len(unique_classes) > 1:
        y_true_bin = label_binarize(y_true.numpy(), classes=[0, 1, 2, 3, 4])
        auc = roc_auc_score(y_true_bin, y_score.numpy(), multi_class='ovr')
        print("Chorea test AUC (multi-class):", auc)
        y_pred = torch.argmax(y_score, dim=1)

        cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=[0,1,2,3,4])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix (Chorea 0–4)")
        plt.tight_layout()
        plt.savefig("hd-chorea-detection/confiusion_matrix_initial_results.png")
        plt.show()
    else:
        print("AUC not computed: only one class in test set.")

if __name__ == '__main__':
    main()