import os
import platform
import mne
from mne_bids import BIDSPath, write_raw_bids, write_meg_calibration, write_meg_crosstalk
import numpy as np
import os
import json
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

# # Original paths
# path_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/raw"
# path_json_file = '/Users/elyestabbane/Documents/UNICOG/2-Experiments/replayseq/1-Scripts/analysis_scripts/modules/objects/bad_channels.json'

# Path to BIDS directory
root_path="/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data"
path_BIDS =os.path.join(root_path,"BIDS")
path_raw=os.path.join(root_path,"raw")
path_reports=os.path.join(path_BIDS,'reports')

# Check if the OS is Linux
if platform.system() == 'Linux':
    root_path = "/neurospin/meg/meg_tmp/2024_ReplaySeq_Elyes/REPLAYSEQ/2-Data/raw"


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
