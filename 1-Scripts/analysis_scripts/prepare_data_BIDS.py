from modules.params import *
from modules.bids_functions import *
import os
import json
import numpy as np
import mne
from mne_bids import write_raw_bids, write_meg_calibration, write_meg_crosstalk, BIDSPath
from matplotlib import pyplot as plt

event_dict={'fixation_blue': 4,
'fixation': 5,
'SequenceID-Rep2/Position-1': 6,
'SequenceID-Rep2/Position-2': 7,
'SequenceID-Rep2/Position-3': 8,
'SequenceID-Rep2/Position-4': 9,
'SequenceID-Rep2/Position-5': 10,
'SequenceID-Rep2/Position-6': 11,
'SequenceID-CRep2/Position-1': 12,
'SequenceID-CRep2/Position-2': 13,
'SequenceID-CRep2/Position-3': 14,
'SequenceID-CRep2/Position-4': 15,
'SequenceID-CRep2/Position-5': 16,
'SequenceID-CRep2/Position-6': 17,
'SequenceID-Rep3/Position-1': 18,
'SequenceID-Rep3/Position-2': 19,
'SequenceID-Rep3/Position-3': 20,
'SequenceID-Rep3/Position-4': 21,
'SequenceID-Rep3/Position-5': 22,
'SequenceID-Rep3/Position-6': 23,
'SequenceID-CRep3/Position-1': 24,
'SequenceID-CRep3/Position-2': 25,
'SequenceID-CRep3/Position-3': 26,
'SequenceID-CRep3/Position-4': 27,
'SequenceID-CRep3/Position-5': 28,
'SequenceID-CRep3/Position-6': 29,
'SequenceID-Rep4/Position-1': 30,
'SequenceID-Rep4/Position-2': 31,
'SequenceID-Rep4/Position-3': 32,
'SequenceID-Rep4/Position-4': 33,
'SequenceID-Rep4/Position-5': 34,
'SequenceID-Rep4/Position-6': 35,
'SequenceID-CRep4/Position-1': 36,
'SequenceID-CRep4/Position-2': 37,
'SequenceID-CRep4/Position-3': 38,
'SequenceID-CRep4/Position-4': 39,
'SequenceID-CRep4/Position-5': 40,
'SequenceID-CRep4/Position-6': 41,
'SequenceID-RepEmbed/Position-1': 42,
'SequenceID-RepEmbed/Position-2': 43,
'SequenceID-RepEmbed/Position-3': 44,
'SequenceID-RepEmbed/Position-4': 45,
'SequenceID-RepEmbed/Position-5': 46,
'SequenceID-RepEmbed/Position-6': 47,
'SequenceID-C1RepEmbed/Position-1': 48,
'SequenceID-C1RepEmbed/Position-2': 49,
'SequenceID-C1RepEmbed/Position-3': 50,
'SequenceID-C1RepEmbed/Position-4': 51,
'SequenceID-C1RepEmbed/Position-5': 52,
'SequenceID-C1RepEmbed/Position-6': 53,
'SequenceID-C2RepEmbed/Position-1': 54,
'SequenceID-C2RepEmbed/Position-2': 55,
'SequenceID-C2RepEmbed/Position-3': 56,
'SequenceID-C2RepEmbed/Position-4': 57,
'SequenceID-C2RepEmbed/Position-5': 58,
'SequenceID-C2RepEmbed/Position-6': 59,
'win': 60,
'loss': 61,
'BAD_muscle':62}



# Main Code
sub_list = range(4,18)
path_bad_channels = "/Users/elyestabbane/Documents/UNICOG/2-Experiments/replayseq/1-Scripts/analysis_scripts/modules/objects/bad_channels.json"


with open(path_bad_channels, 'r') as file:
    bad_channels_dict = json.load(file)

for sub in sub_list:
    for run in range(1, 19):
        lab = bad_channels_dict[f"sub-{sub:02}"]['lab']
        original_data_path = os.path.join(path_raw, "Data_ICM" if lab == 'icm' else f'Data_neurospin')
        data_path = os.path.join(original_data_path, f"sub-{sub:02}/run{run:02}_raw.fif")

        print(f"--- Subject-{sub:02} ({lab}): saving in bids format run-{run} ------")
        raw = mne.io.read_raw_fif(data_path, allow_maxshield=True, preload=True)

        # Add the events
        print('Adding Events')
        n_chans = raw.get_data().shape[0]
        raw.pick(np.where(['EEG' not in raw.info['ch_names'][i] for i in range(n_chans)])[0])

        if lab == 'icm':
            raw.set_channel_types({'BIO001': 'eog'})
            raw.set_channel_types({'BIO002': 'eog'})
            raw.set_channel_types({'BIO003': 'ecg'})
            events, event_ids = extract_events_and_event_IDs_ICM(raw)
        else:
            events, event_ids = extract_events_and_event_IDs_neurospin(raw)

        # Adding the bad_muscle id to the event_ids
        event_ids['BAD_muscle']=event_dict['BAD_muscle']

        bids_path = BIDSPath(subject=f'{sub:02}', task='reproduction', run=run, datatype='meg', root=path_BIDS)

        # Add the bad channels
        raw.info['bads'] = bad_channels_dict[f"sub-{sub:02}"][f"run{run:02}"]


        # Annotate the muscle artifacts
        print('Annotating muscle artifact')
        annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
            raw,
            ch_type="mag",
            threshold=5,
            min_length_good=0.2,
            filter_freq=[100, 300],
        )
        raw.set_annotations(annot_muscle)

        # Save the data
        write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='FIF', events=events, event_id=event_ids, overwrite=True)
        print(f'File written in {bids_path}')

        # Write MEG calibration files
        print('writing MEG calibration files')
        if lab == 'icm':
            ct_fname = path_BIDS + "/calibration_files/calibration_icm/ct_sparse.fif"
            cal_fname = path_BIDS + "/calibration_files/calibration_icm/sss_cal_3101_160108.dat"
        else:
            ct_fname = path_BIDS + "/calibration_files/calibration_neurospin/ct_sparse.fif"
            cal_fname = path_BIDS + "/calibration_files/calibration_neurospin/sss_cal_3176_20240123_2.dat"

        write_meg_calibration(calibration=cal_fname, bids_path=bids_path)
        write_meg_crosstalk(fname=ct_fname, bids_path=bids_path)

        # Generate the BIDS report
        generate_bids_report(sub, run, raw, annot_muscle, bids_path, path_reports)
