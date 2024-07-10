import mne
from mne_bids import BIDSPath, write_raw_bids, write_meg_calibration, write_meg_crosstalk
import numpy as np
import os
# from params import *

path_root="/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/raw"


def extract_events_and_event_IDs(raw):
    """
    The trigger codes are
    9 for fixation,
    trigger_seq + item + 1 for the items of the sequence,
    99 at the beginning of the reproduction phase and
    128 at the beginning of the resting state phase.
    """

    events = mne.find_events(raw, min_duration=0.01)
    # on élimine les évènements de feedback de score (10*DL distance)
    events = events[events[:,2]%10!=0,:]
    events_fixation = events[events[:,2]==9,:]
    events_sequence_presentation = events[[events[i,2]>10 and events[i,2]<97 for i in range(len(events))]]
    events_fixation_blue = events[events[:,2]==99]
    events_resting_phase = events[events[:,2]==128]
    events_of_interest = np.vstack([events_fixation,events_fixation_blue,events_sequence_presentation,events_resting_phase])

    dict_fixation = {'Fixation':9}
    dict_reproduction = {'Reproduction':99}
    seqID = int(np.floor(events_sequence_presentation[0,2]/10)*10)
    sequence_events = {10: 'Rep2',20: 'CRep2',30: 'Rep3',40: 'CRep3',50: 'Rep4',60: 'CRep4',70: 'RepEmbed',80: 'C1RepEmbed',90: 'C2RepEmbed'}
    dict_sequences = {'SequenceID-%s/Position-%i'%(sequence_events[seqID],i):seqID+i  for i in range(1,7)}
    dict_resting_state = {'Resting_state':128}
    event_ids_dict =  dict_fixation | dict_reproduction | dict_sequences | dict_resting_state

    return events_of_interest, event_ids_dict



def prepare_data_for_mne_bids_pipeline(subject='sub-01',base_path = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ",icm=False):

    original_data_path = base_path + "/1-data_ICM/raw/"
    root = base_path+'/1-data_ICM/BIDS'

    for run in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']:
        print("--- saving in bids format run %s ------")
        data_path = original_data_path+subject+ '/run%s.fif'%run
        raw = mne.io.read_raw_fif(data_path,allow_maxshield=True,preload=True)
        n_chans=raw.get_data().shape[0]
        raw.pick(np.where(['EEG' not in raw.info['ch_names'][i] for i in range(n_chans)])[0])
        
        if icm:
            raw.set_channel_types({'BIO002': 'eog'})
            raw.set_channel_types({'BIO003': 'ecg'})
        
        events, event_ids = extract_events_and_event_IDs(raw)
        bids_path = BIDSPath(subject=subject[-2:], task='reproduction', run=run, datatype='meg', root=root)

        write_raw_bids(raw, bids_path=bids_path,allow_preload=True,format='FIF',events_data=events,event_id=event_ids,overwrite=True)

        # write MEG calibration files
        cal_fname = root+'/system_calibration_files/sss_cal_3101_160108.dat'
        ct_fname = root+'/system_calibration_files/ct_sparse.fif'

        write_meg_calibration(calibration=cal_fname,bids_path=bids_path)
        write_meg_crosstalk(fname=ct_fname,bids_path=bids_path)

def inspect_raw(sub_nb, run, path_root=path_root,icm=False):
    
    if icm:
        path_raw=os.path.join(path_root,'Data_ICM')
    else:
        path_raw=os.path.join(path_root,'Data_neurospin')
        
    path_raw_file=os.path.join(path_raw,f'sub-{sub_nb:02}/run{run:02}.fif')
    
    # Open the raw object
    raw=mne.io.read_raw_fif(path_raw_file, allow_maxshield=True, preload=True)
    
    # 1 - Plot the raw object for the given subjet / run.
    raw.plot()
    
    # 2 - Plot the PSD to note outliers 
    raw.compute_psd().plot()