import mne

study_name = "REPLAYSEQ"

bids_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/1-data_ICM/BIDS"
deriv_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/1-data_ICM/3-epochs_sequences/mne-bids-pipeline/"

task = "reproduction"

runs = 'all'

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = bids_root + "/system_calibration_files/ct_sparse.fif"
mf_cal_fname = bids_root + "/system_calibration_files/sss_cal_3101_160108.dat"

# ch_types = ["meg","eeg"] # No numerization points for EEG so it outputs an error.
ch_types = ["meg"]
data_type='meg'

# Trying to give a standard positionning of eeg channels
#eeg_template_montage = mne.channels.make_standard_montage("standard_1005")
#eeg_template_montage = "easycap-M10"

l_freq = None
h_freq = 40.0


# SSP and peak-to-peak rejection
spatial_filter = "ica" #NOTE DIFF
n_proj_eog = dict(n_mag=0, n_grad=0)
n_proj_ecg = dict(n_mag=2, n_grad=2)
reject = {"grad": 4000e-13, "mag": 4e-12} #NOTE DIFF

# Epochs
epochs_tmin = -0.2
epochs_tmax = 1 + 0.4*12+6+0.5  #durée de la croix de fixation + SOA * nb_items + durée de pause (compression time) + un petit délais (pour avoir fixation bleue)
epochs_decim = 4
baseline = None

# Conditions / events to consider when epoching
#conditions = ['SequenceID-Rep2/Position-1', 'SequenceID-Rep2/Position-2', 'SequenceID-Rep2/Position-3', 'SequenceID-Rep2/Position-4', 'SequenceID-Rep2/Position-5', 'SequenceID-Rep2/Position-6', 'SequenceID-CRep2/Position-1', 'SequenceID-CRep2/Position-2', 'SequenceID-CRep2/Position-3', 'SequenceID-CRep2/Position-4', 'SequenceID-CRep2/Position-5', 'SequenceID-CRep2/Position-6', 'SequenceID-Rep3/Position-1', 'SequenceID-Rep3/Position-2', 'SequenceID-Rep3/Position-3', 'SequenceID-Rep3/Position-4', 'SequenceID-Rep3/Position-5', 'SequenceID-Rep3/Position-6', 'SequenceID-CRep3/Position-1', 'SequenceID-CRep3/Position-2', 'SequenceID-CRep3/Position-3', 'SequenceID-CRep3/Position-4', 'SequenceID-CRep3/Position-5', 'SequenceID-CRep3/Position-6', 'SequenceID-Rep4/Position-1', 'SequenceID-Rep4/Position-2', 'SequenceID-Rep4/Position-3', 'SequenceID-Rep4/Position-4', 'SequenceID-Rep4/Position-5', 'SequenceID-Rep4/Position-6', 'SequenceID-CRep4/Position-1', 'SequenceID-CRep4/Position-2', 'SequenceID-CRep4/Position-3', 'SequenceID-CRep4/Position-4', 'SequenceID-CRep4/Position-5', 'SequenceID-CRep4/Position-6', 'SequenceID-RepEmbed/Position-1', 'SequenceID-RepEmbed/Position-2', 'SequenceID-RepEmbed/Position-3', 'SequenceID-RepEmbed/Position-4', 'SequenceID-RepEmbed/Position-5', 'SequenceID-RepEmbed/Position-6', 'SequenceID-C1RepEmbed/Position-1', 'SequenceID-C1RepEmbed/Position-2', 'SequenceID-C1RepEmbed/Position-3', 'SequenceID-C1RepEmbed/Position-4', 'SequenceID-C1RepEmbed/Position-5', 'SequenceID-C1RepEmbed/Position-6', 'SequenceID-C2RepEmbed/Position-1', 'SequenceID-C2RepEmbed/Position-2', 'SequenceID-C2RepEmbed/Position-3', 'SequenceID-C2RepEmbed/Position-4', 'SequenceID-C2RepEmbed/Position-5', 'SequenceID-C2RepEmbed/Position-6']
"""
    TODO Je sélectionne la croix de fixation (qui indique le début pour toutes les séquences). Mais je n'aurais plus l'information sur quelle séquence a été montrée
    donc je vais devoir utiliser le csv des stims.
    Je peux m'aider de stimulation_scripts>config.py : avoir toutes les informations sur les paramètres expérimentaux 
"""
conditions=['Fixation']

# This is often helpful when doing multiple subjects.  If 1 subject fails processing stops
#on_error = 'continue'

# Decoding
decode = False

# Noise estimation
process_empty_room = True #NOTE DIFF


# TODO : mettre ICA à la place. Voir les configs files sur le site de mne_bids_pipeline