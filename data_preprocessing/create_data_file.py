import pandas as pd

def process_file(file_path):
    """Process the CSV file and return walking and chorea ranges."""
    df = pd.read_csv(file_path, header=None, sep="\t", engine="python", quoting=3)
    df = df[0].str.split(',', expand=True)

    walking_ranges = []
    chorea_ranges = []

    for _, row in df.iterrows():
        column = str(row[4]).lower()  # Convert the 5th column to lowercase string
        start_frame, end_frame = pd.to_numeric([row[2], row[3]], errors='coerce')

        if "walking" in column:
            walking_ranges.append((start_frame, end_frame))

        try:
            label = int(column.split()[0])
            if 0 <= label < 5:
                chorea_ranges.append((start_frame, end_frame, label))
        except:
            pass

    return walking_ranges, chorea_ranges

def find_chorea_label(frame, chorea_ranges):
    """Find the chorea label for a given frame."""
    for start, end, label in chorea_ranges:
        if start <= frame <= end:
            return label
    return -9 

def generate_walking_frames_data(walking_ranges, chorea_ranges):
    """Generate walking frames data with chorea labels."""
    walking_frames_data = []
    for start, end in walking_ranges:
        for frame in range(start, end + 1):  # Iterate through all frames in the walking range
            chorea_label = find_chorea_label(frame, chorea_ranges)
            walking_frames_data.append([frame, chorea_label])
    return walking_frames_data

def compare_files(df1, df2, output_comparison_path):
    """Compare two walking frames dataframes and save mismatches."""
    frames_file1 = set(df1["Frame"])
    frames_file2 = set(df2["Frame"])

    # Find frames in file1 but not in file2
    missing_in_file2 = frames_file1 - frames_file2
    # Find frames in file2 but not in file1
    missing_in_file1 = frames_file2 - frames_file1

    mismatches = []

    # Check frames that are in file1 but not in file2
    for frame in missing_in_file2:
        mismatches.append({
            "Frame": frame,
            "Chorea Label File 1": df1[df1["Frame"] == frame]["Chorea Label"].values[0],
            "Chorea Label File 2": -9,
            "Issue": "Missing Frame in File 2"
        })

    # Check frames that are in file2 but not in file1
    for frame in missing_in_file1:
        mismatches.append({
            "Frame": frame,
            "Chorea Label File 1": -9,
            "Chorea Label File 2": df2[df2["Frame"] == frame]["Chorea Label"].values[0],
            "Issue": "Missing Frame in File 1"
        })

    # Check frames that are in both files but have different chorea labels
    common_frames = frames_file1 & frames_file2
    for frame in common_frames:
        label1 = df1[df1["Frame"] == frame]["Chorea Label"].values[0]
        label2 = df2[df2["Frame"] == frame]["Chorea Label"].values[0]
        if label1 != label2:
            mismatches.append({
                "Frame": frame,
                "Chorea Label File 1": label1,
                "Chorea Label File 2": label2,
                "Issue": "Mismatch"
            })

    # Convert to a DataFrame and sort by frame
    mismatches_df = pd.DataFrame(mismatches)
    mismatches_df = mismatches_df.sort_values(by="Frame").reset_index(drop=True)

    # Save to CSV
    mismatches_df.to_csv(output_comparison_path, index=False)
    print(f"Comparison complete. Mismatches saved to '{output_comparison_path}'.")

# File paths and output paths
file1_path = r"/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/timeline_NB.csv"
file2_path = r"/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/timeline_Gh.csv"
output1_path = "/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/walking_frames_with_chorea_10TC_NB.csv"
output2_path = "/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/walking_frames_with_chorea_10TC_GH.csv"
output_comparison_path = "/home/netabiran/hd-chorea-detection/data_preprocessing/data/10TC/comparison_mismatches_10TC.csv"

# Process the first file
walking_ranges1, chorea_ranges1 = process_file(file1_path)
walking_frames_data1 = generate_walking_frames_data(walking_ranges1, chorea_ranges1)

# Process the second file
walking_ranges2, chorea_ranges2 = process_file(file2_path)
walking_frames_data2 = generate_walking_frames_data(walking_ranges2, chorea_ranges2)

# Save the walking frames data to CSV
pd.DataFrame(walking_frames_data1, columns=["Frame", "Chorea Label"]).to_csv(output1_path, index=False)
pd.DataFrame(walking_frames_data2, columns=["Frame", "Chorea Label"]).to_csv(output2_path, index=False)

# Compare the two dataframes and save mismatches
compare_files(pd.DataFrame(walking_frames_data1, columns=["Frame", "Chorea Label"]), 
              pd.DataFrame(walking_frames_data2, columns=["Frame", "Chorea Label"]), 
              output_comparison_path)