import mne

path_root = "/neurospin/meg/meg_tmp/2024_ReplaySeq_Elyes/REPLAYSEQ/2-Data/"

study_name = "REPLAYSEQ"

bids_root = path_root+"BIDS"
deriv_root = path_root+"derivatives/items"

task = "reproduction"

runs = 'all'
exclude_subjects = ['01','02']
exclude_subjects.extend(['04', '05']) # subject that don't have bad channels annotated yet

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
'Rep2-1',
 'Rep2-2',
 'Rep2-3',
 'Rep2-4',
 'Rep2-5',
 'Rep2-6',
 'CRep2-1',
 'CRep2-2',
 'CRep2-3',
 'CRep2-4',
 'CRep2-5',
 'CRep2-6',
 'Rep3-1',
 'Rep3-2',
 'Rep3-3',
 'Rep3-4',
 'Rep3-5',
 'Rep3-6',
 'CRep3-1',
 'CRep3-2',
 'CRep3-3',
 'CRep3-4',
 'CRep3-5',
 'CRep3-6',
 'Rep4-1',
 'Rep4-2',
 'Rep4-3',
 'Rep4-4',
 'Rep4-5',
 'Rep4-6',
 'CRep4-1',
 'CRep4-2',
 'CRep4-3',
 'CRep4-4',
 'CRep4-5',
 'CRep4-6',
 'RepEmbed-1',
 'RepEmbed-2',
 'RepEmbed-3',
 'RepEmbed-4',
 'RepEmbed-5',
 'RepEmbed-6',
 'C1RepEmbed-1',
 'C1RepEmbed-2',
 'C1RepEmbed-3',
 'C1RepEmbed-4',
 'C1RepEmbed-5',
 'C1RepEmbed-6',
 'C2RepEmbed-1',
 'C2RepEmbed-2',
 'C2RepEmbed-3',
 'C2RepEmbed-4',
 'C2RepEmbed-5',
 'C2RepEmbed-6']

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