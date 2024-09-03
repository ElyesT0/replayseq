import mne


# Boolean that indicates if script is ran on neurospin server or personnal computer
server=False
hard_drive=False
id_pc="pc_id"

if server:
    path_root = "/neurospin/meg/meg_tmp/2024_ReplaySeq_Elyes/replayseq/2-Data/"
else:
    if hard_drive:
        path_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/"
    else:
        path_root=f"/volatile/home/{id_pc}/Documents/replayseq/2-Data/"
        

study_name = "REPLAYSEQ"

bids_root = path_root+"BIDS"
deriv_root = path_root+"derivatives/items"

task = "reproduction"

runs = 'all'
exclude_subjects = []
exclude_subjects.extend(['08']) # subject that don't have bad channels annotated yet

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = bids_root + "/calibration_files/calibration_neurospin/ct_sparse.fif"
mf_cal_fname = bids_root + "/calibration_files/calibration_neurospin/sss_cal_3176_20240123_2.dat"

# ch_types = ["meg","eeg"] # No numerization points for EEG so it outputs an error.
ch_types = ["meg"]
data_type='meg'


# Trying to give a standard positionning of eeg channels
#eeg_template_montage = mne.channels.make_standard_montage("standard_1005")
# eeg_template_montage = "easycap-M10"

raw_resample_sfreq=250
l_freq = None
h_freq = 40.0


ica_l_freq = 1.0
spatial_filter = "ica"
ica_reject = {"grad": 4000e-13, "mag": 4e-12} 
ica_max_iterations = 1000
ica_algorithm="fastica"
ica_n_components=0.99

reject="autoreject_global"

# Epochs
epochs_tmin = -0.2
epochs_tmax = 0.6
epochs_decim = 4
baseline = (-0.2,0)

# Conditions / events to consider when epoching
conditions = [
'SequenceID-Rep2/Position-1',
'SequenceID-Rep2/Position-2',
'SequenceID-Rep2/Position-3',
'SequenceID-Rep2/Position-4',
'SequenceID-Rep2/Position-5',
'SequenceID-Rep2/Position-6',
'SequenceID-CRep2/Position-1',
'SequenceID-CRep2/Position-2',
'SequenceID-CRep2/Position-3',
'SequenceID-CRep2/Position-4',
'SequenceID-CRep2/Position-5',
'SequenceID-CRep2/Position-6',
'SequenceID-Rep3/Position-1',
'SequenceID-Rep3/Position-2',
'SequenceID-Rep3/Position-3',
'SequenceID-Rep3/Position-4',
'SequenceID-Rep3/Position-5',
'SequenceID-Rep3/Position-6',
'SequenceID-CRep3/Position-1',
'SequenceID-CRep3/Position-2',
'SequenceID-CRep3/Position-3',
'SequenceID-CRep3/Position-4',
'SequenceID-CRep3/Position-5',
'SequenceID-CRep3/Position-6',
'SequenceID-Rep4/Position-1',
'SequenceID-Rep4/Position-2',
'SequenceID-Rep4/Position-3',
'SequenceID-Rep4/Position-4',
'SequenceID-Rep4/Position-5',
'SequenceID-Rep4/Position-6',
'SequenceID-CRep4/Position-1',
'SequenceID-CRep4/Position-2',
'SequenceID-CRep4/Position-3',
'SequenceID-CRep4/Position-4',
'SequenceID-CRep4/Position-5',
'SequenceID-CRep4/Position-6',
'SequenceID-RepEmbed/Position-1',
'SequenceID-RepEmbed/Position-2',
'SequenceID-RepEmbed/Position-3',
'SequenceID-RepEmbed/Position-4',
'SequenceID-RepEmbed/Position-5',
'SequenceID-RepEmbed/Position-6',
'SequenceID-C1RepEmbed/Position-1',
'SequenceID-C1RepEmbed/Position-2',
'SequenceID-C1RepEmbed/Position-3',
'SequenceID-C1RepEmbed/Position-4',
'SequenceID-C1RepEmbed/Position-5',
'SequenceID-C1RepEmbed/Position-6',
'SequenceID-C2RepEmbed/Position-1',
'SequenceID-C2RepEmbed/Position-2',
'SequenceID-C2RepEmbed/Position-3',
'SequenceID-C2RepEmbed/Position-4',
'SequenceID-C2RepEmbed/Position-5',
'SequenceID-C2RepEmbed/Position-6']

# This is often helpful when doing multiple subjects.  If 1 subject fails processing stops
#on_error = 'continue'

# Decoding
decode = False

# Noise estimation
process_empty_room = True

# noise_cov == None
ssp_meg = "combined"

# Configuration to apply ICA and the new reject parameter
# This block would typically be run by the MNE-BIDS pipeline

# Channel re-tagging dictionary
# channel_retagging = {
#     'BIO002': 'eog',
#     'BIO003': 'ecg'
# }

# def preprocess_and_apply_ica(raw):
#     # Apply channel re-tagging
#     raw.set_channel_types(channel_retagging)
    
#     # Apply ICA
#     ica = mne.preprocessing.ICA(n_components=ica_n_components, reject=ica_reject)
#     ica.fit(raw)
    
#     # Optionally save the ICA solution if needed
#     # ica.save(f"{deriv_root}/ica_solution.fif")
    
#     return raw, ica