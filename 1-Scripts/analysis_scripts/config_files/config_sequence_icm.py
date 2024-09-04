import mne


# Boolean that indicates if script is ran on neurospin server or personnal computer
server=False
hard_drive=True
pc_id="pc_id"

if server:
    path_root = "/neurospin/meg/meg_tmp/2024_ReplaySeq_Elyes/replayseq/2-Data/"
else:
    if hard_drive:
        path_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/"
    else:
        path_root=f"/volatile/home/{pc_id}/Documents/replayseq/2-Data/"

study_name = "REPLAYSEQ"

bids_root = path_root+"BIDS"
deriv_root = path_root+"derivatives/sequence"

task = "reproduction"

runs = 'all'
exclude_subjects = ['03','04','05','06','07','08','09']
exclude_subjects.extend([]) # subject that don't have bad channels annotated yet

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = bids_root + "/calibration_files/calibration_icm/ct_sparse.fif"
mf_cal_fname = bids_root + "/calibration_files/calibration_icm/sss_cal_3101_160108.dat"

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
epochs_tmax = 1 + 0.4*12+6+0.5  #durée de la croix de fixation + SOA * nb_items + durée de pause (compression time) + un petit délais (pour avoir fixation bleue)
epochs_decim = 1
baseline = (None,0)


# Conditions / events to consider when epoching
conditions = ['fixation']

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