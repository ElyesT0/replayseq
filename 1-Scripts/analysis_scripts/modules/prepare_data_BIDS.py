import os
import mne
from mne_bids import BIDSPath, write_raw_bids, write_meg_calibration, write_meg_crosstalk
from params import *
import json
import numpy as np

# Path to BIDS directory
path_bids = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/BIDS"

def process_subject(sub, path_bids):
    # Load JSON file for bad channels
    with open(path_json_file, 'r') as file:
        bad_channels_dict = json.load(file)

    subject = f'sub-{sub:02}'
    lab = bad_channels_dict[subject]['lab']

    if lab == 'icm':
        original_data_path = os.path.join(path_root, 'Data_ICM', subject)
    else:
        original_data_path = os.path.join(path_root, 'Data_neurospin', subject)

    print(f"Processing participant {subject}")

    for run in range(1, 19):
        run_str = f'run{run:02}'
        if lab == 'icm':
            data_path = os.path.join(original_data_path, f'{run_str}.fif')
        else:
            data_path = os.path.join(original_data_path, f'{run_str}_raw.fif')

        if not os.path.exists(data_path):
            print(f"File not found: {data_path}, skipping run {run_str}")
            continue

        try:
            # Load raw data
            raw = mne.io.read_raw_fif(data_path, allow_maxshield=True, preload=True, verbose=False)

            # Annotate muscle artifacts
            annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
                raw,
                ch_type="mag",
                threshold=5,
                min_length_good=0.2,
                filter_freq=[100, 300],
            )

            # Set annotations
            raw.set_annotations(annot_muscle)

            # Remove muscle artifacts
            raw_cleaned = raw.copy().drop_bad(reject_by_annotation=True)

            # Update bad channels from JSON file
            raw_cleaned.info['bads'] = bad_channels_dict[subject][run_str]

            # Extract events and event IDs
            if lab == 'icm':
                events = mne.find_events(raw_cleaned, min_duration=0.01, verbose=False)
                events = events[events[:, 2] % 10 != 0, :]
                event_ids = {"Fixation": 9, "Reproduction": 99, "Resting_state": 128}
            else:
                events = mne.find_events(raw_cleaned, mask_type="not_and", mask=2**6 + 2**7, verbose=False)
                event_ids = event_dict

            # Create BIDS path
            bids_path = BIDSPath(
                subject=f'{sub:02}',
                task='reproduction',
                run=run,
                datatype='meg',
                root=path_bids
            )

            # Save cleaned data in BIDS format
            write_raw_bids(
                raw_cleaned,
                bids_path=bids_path,
                events=events,
                event_id=event_ids,
                overwrite=True,
                format='FIF',
                verbose=False
            )

            # Save MEG calibration and crosstalk files
            if lab == 'icm':
                ct_fname = os.path.join(path_bids, "calibration_files/calibration_icm/ct_sparse.fif")
                cal_fname = os.path.join(path_bids, "calibration_files/calibration_icm/sss_cal_3101_160108.dat")
            else:
                ct_fname = os.path.join(path_bids, "calibration_files/calibration_neurospin/ct_sparse.fif")
                cal_fname = os.path.join(path_bids, "calibration_files/calibration_neurospin/sss_cal_3176_20240123_2.dat")

            write_meg_calibration(calibration=cal_fname, bids_path=bids_path, verbose=False)
            write_meg_crosstalk(fname=ct_fname, bids_path=bids_path, verbose=False)

            # Plot power spectral density (PSD) and save the figure
            fig = raw_cleaned.compute_psd(picks='mag', reject_by_annotation=True).plot()
            fig.savefig(os.path.join(path_bids, f'sub-{sub:02}_run{run:02}_psd.png'))

        except Exception as e:
            print(f"An error occurred for subject {sub}, run {run_str}: {e}")

    print(f"Saving {subject} to {path_bids}")

# List of subjects
subjects = [i for i in range(1, 19)]

# Process each subject
for sub in subjects:
    process_subject(sub, path_bids)

print("Processing complete.")
