import os
import numpy as np
import pandas as pd
import csv
import ipdb

# Constants
SYNC_FILE = "/home/netabiran/data_ready/sync_params.xlsx"
ACC_DIR = "/home/dafnas1/datasets/hd_dataset/lab_geneactive/acc_data/right_wrist"
LABEL_DIR = "/home/netabiran/data_ready/hd_dataset/lab_geneactive/labeled_data"
TARGET_DIR = "/home/netabiran/data_ready/hd_dataset/lab_geneactive/synced_labeled_data"

KEY_COLUMN = "video name"
VALUE_COLUMNS = ["video 2m walk start time (seconds)", "sensor 2m walk start time (seconds)", "FPS"]
SYNC_SHEET = "Sheet1"
ACC_SAMPLE_RATE = 100  # Hz

ACTIVITY_DICT = {
    'walking': 1, 'moving (small steps)': -9, 'turning around': 0, 'turning ': 0, 'stepping': -9, 'stumbling': 0, 
    'stepping to the side': 0, 'standing': 0, 'sitting': 0, 'sitting down': 0, 'standing clapping hands': 0, 
    'sitting clapping hands': 0, 'sit to stand': 0, 'sitting and writing': 0, 'sitting and drinking water': 0, 
    'standing up': 0, 'standing and clapping hands': 0, 'standing and putting arms crossed on the chest': 0,
    'standing with arms crossed on the chest': 0, 'standing and putting the arms down': 0, 
    'standing and putting the hands down': 0, 'stepping on a foam': 0, 'stepping off the foam': 0, 
    'bending over': 0, 'stepping up and down a step': 0, 'moving hands up': 0, 'clapping hands': 0, 
    'moving hands down': 0, 'stumbling': 0, 'stepping over a step': -9, 'stepping off a step': -9, 'turning around': -9, 
    'walking backwards': -9, 'going down stairs': -9, 'climbing up stairs': -9, 'stepping backward': -9, 
    'step up': -9, 'step down': 0, 'moving with chair': -9, 'going forward with chair': 0, 'going backward with chair': 0,
    '-9': -9
}

def main(modes=["video"]):
    if "video" in modes:
        sync_data_dict = load_sync_data(SYNC_FILE, SYNC_SHEET, KEY_COLUMN, VALUE_COLUMNS)

        for patient, (video_start, sensor_start, label_rate) in sync_data_dict.items():
            acc_data = load_accelerometer_data(patient)
            label_data, chorea_labels = load_label_data(patient, label_rate, ACC_SAMPLE_RATE)

            if label_data is None:
                continue

            synced_data = synchronize_data(
                acc_data, label_data, chorea_labels, (video_start, sensor_start), ACC_SAMPLE_RATE
            )

            save_data(patient, *synced_data)


def load_sync_data(file_path, sheet_name, key_col, value_cols):
    """Loads synchronization parameters from an Excel sheet."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    return dict(zip(df[key_col].str.split("_").str[0], df[value_cols].apply(tuple, axis=1)))


def load_accelerometer_data(patient, directory=ACC_DIR):
    """Loads accelerometer data for a given patient."""
    file_name = next((f for f in os.listdir(directory) if f.startswith(f"{patient}_")), None)

    if file_name is None:
        print(f"No accelerometer data found for {patient}")
        return None

    print(f"Processing {file_name}")
    file_path = os.path.join(directory, file_name)

    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader((line.replace("\0", "") for line in file))
        next(csv_reader)  # Skip header
        for row in csv_reader:
            if len(row) >= 4:
                data.append([row[1], row[2], row[3]])

    return np.array(data) if data else None


def load_label_data(patient, source_sample_rate, target_sample_rate=ACC_SAMPLE_RATE, files_dir=LABEL_DIR):
    """Loads label data and synchronizes it with the accelerometer sample rate."""
    
    # Step 1: Find label file
    timeline_csv_path = find_label_file(patient, files_dir)
    if timeline_csv_path is None:
        print(f'No matching labels file or folder exists for {patient}')
        return None, None

    # Step 2: Parse the label file
    labels_by_frames, chorea_labels_by_frames = parse_labels(timeline_csv_path)
    if not labels_by_frames:
        print(f'Patient {patient} has no labels.')
        return None, None

    # Step 3: Create label arrays
    try:
        last_labeled_frame = labels_by_frames[-1][1]
    except Exception as e:
        print(f"Error extracting last labeled frame: {e}")
        ipdb.set_trace()

    last_labeled_frame_sample = int(np.round(last_labeled_frame * (target_sample_rate / source_sample_rate)))
    labels_array = -np.ones(last_labeled_frame_sample + 1, dtype=int)

    # Step 4: Map frame-based labels to sample-based labels
    for label_set in labels_by_frames:
        activity_name = label_set[2]
        if activity_name not in ACTIVITY_DICT:
            print(f"Warning: Label '{activity_name}' not found for patient {patient}")
            continue
        start_sample = int(np.round(label_set[0]) * (target_sample_rate / source_sample_rate))
        end_sample = int(np.round(label_set[1]) * (target_sample_rate / source_sample_rate))
        labels_array[start_sample:end_sample] = ACTIVITY_DICT.get(activity_name, -1)

    # Step 5: Handle chorea labels
    chorea_labels = np.ones(last_labeled_frame_sample + 1, dtype=int) * -1
    for label_set in chorea_labels_by_frames:
        start_sample = int(np.round(label_set[0]) * (target_sample_rate / source_sample_rate))
        end_sample = int(np.round(label_set[1]) * (target_sample_rate / source_sample_rate))

        if label_set[2] in ['', 'hided', '-9']:
            level = -1
        else:
            try:
                level = int(label_set[2])
            except (ValueError, TypeError):
                level = -1

        chorea_labels[start_sample:end_sample + 1] = level

    return labels_array, chorea_labels


def find_label_file(patient, files_dir):
    """Find the label file for the patient."""
    subfolder_path = None
    timeline_csv_path = None

    # Search for a subfolder starting with patient ID
    for root, dirs, files in os.walk(files_dir):
        for dir_name in dirs:
            if dir_name.startswith(patient + '_'):
                subfolder_path = os.path.join(root, dir_name)
                break
        if subfolder_path:
            break

    if subfolder_path:
        timeline_csv_path = os.path.join(subfolder_path, "timeline.csv")
    else:
        # If no subfolder, search for matching files directly
        for file in os.listdir(files_dir):
            if file.startswith(patient + '_'):
                timeline_csv_path = os.path.join(files_dir, file)
                break

    return timeline_csv_path


def parse_labels(timeline_csv_path):
    """Parses the 'timeline.csv' to extract label information."""
    labels_by_frames = []
    chorea_labels_by_frames = []
    sections_counter = 0

    try:
        with open(timeline_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if not row or all(cell == '' for cell in row):
                    continue
                if sections_counter < 3:
                    if row[0] == 'T':
                        sections_counter += 1
                        continue
                if sections_counter == 2:
                    try:
                        first_frame = int(row[2])
                        last_frame = int(row[3])
                        activity_name = row[4]
                        labels_by_frames.append((first_frame, last_frame, activity_name))
                    except ValueError:
                        continue
                if sections_counter == 3:
                    if row[0] == 'T':
                        break
                    try:
                        first_frame = int(row[2])
                        last_frame = int(row[3])
                        activity_name = row[4]
                        chorea_labels_by_frames.append((first_frame, last_frame, activity_name))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error parsing labels: {e}")
        ipdb.set_trace()

    return labels_by_frames, chorea_labels_by_frames


def map_labels_to_samples(labels, chorea_labels, source_rate, target_rate):
    """Maps frame-based labels to sample-based labels with the correct sample rate."""
    activity_mapping = {
        "walking": 1, "moving (small steps)": 0, "turning_around": 0, "stepping": 0, "standing": 0, "sitting": 0,
        "clapping hands": 0, "stepping backwards": -9, "walking backward": -9, "stairs up": -9,
        "step up": -9, "climbing up stairs": -9, "going down stairs": -9, "-9": -9
    }

    last_frame = labels[-1][1]
    last_sample = int(np.round(last_frame * (target_rate / source_rate)))
    label_array = np.full(last_sample + 1, -1, dtype=int)

    for start, end, activity in labels:
        if activity in activity_mapping:
            start_sample = int(np.round(start * (target_rate / source_rate)))
            end_sample = int(np.round(end * (target_rate / source_rate)))
            label_array[start_sample:end_sample] = activity_mapping[activity]

    chorea_array = np.full(last_sample + 1, -1, dtype=int)
    for start, end, activity in chorea_labels:
        start_sample = int(np.round(start * (target_rate / source_rate)))
        end_sample = int(np.round(end * (target_rate / source_rate)))
        chorea_array[start_sample:end_sample] = -1 if activity in ["", "hided", "-9"] else int(activity)

    return label_array, chorea_array


def synchronize_data(acc_data, label_data, chorea_labels, sync_times, sample_rate):
    """Synchronizes accelerometer and label data based on provided sync times."""
    acc_sync_offset = int(sample_rate * sync_times[1])
    label_sync_offset = int(sample_rate * sync_times[0])

    acc_data = acc_data[max(0, acc_sync_offset - label_sync_offset) :]
    label_start = max(0, label_sync_offset - acc_sync_offset)

    label_data = label_data[label_start:]
    chorea_labels = chorea_labels[label_start:]

    acc_data = acc_data[: label_data.shape[0]]
    time_data = np.arange(label_data.shape[0]) / sample_rate + label_start / sample_rate

    return acc_data, label_data, chorea_labels, time_data


def save_data(patient, acc_data, label_data, chorea_labels, time_data):
    """Saves the synchronized data as a NumPy archive."""
    file_path = os.path.join(TARGET_DIR, f"{patient}.npz")
    np.savez(file_path, acc_data=acc_data, label_data=label_data, chorea_labels=chorea_labels, time_data=time_data)
    print(f"Saved synced data for {patient} at {file_path}")


if __name__ == "__main__":
    main(["video"])