import mne
from mne_bids import BIDSPath, write_raw_bids, write_meg_calibration, write_meg_crosstalk
import mne_bids

def extract_events_and_event_IDs(raw_task):

    import numpy as np
    sequence_events = {10: 'Rep2',20: 'CRep2',30: 'Rep3',40: 'CRep3',50: 'Rep4',60: 'CRep4',70: 'RepEmbed',80: 'C1RepEmbed',90: 'C2RepEmbed'}
    events_presentation = mne.find_events(raw_task,min_duration=0.01,mask_type='not_and',mask=2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15+2**16)

    # In some blocs, there may be the issue that the score is 90%, and then the score and the fixation have the same trigger code...
    events_fixation = []
    for j in range(len(events_presentation)):
        if events_presentation[j,2]==9 and events_presentation[j+1,2] !=9:
            events_presentation[j, 2] = 9999
            events_fixation.append(list(events_presentation[j]))

    events_sequence_presentation = events_presentation[[events_presentation[i,2]>10 and events_presentation[i,2]<97 for i in range(len(events_presentation))]]
    events_feedback_score = events_presentation[events_presentation[:,2]<11]
    events_of_interest = np.vstack([events_fixation,events_sequence_presentation,events_feedback_score])

    dict_scores = {'Score_%i'% int(i): i  for i in np.unique(events_feedback_score[:, 2])}
    dict_fixation = {'Fixation':9999}
    seqID = int(np.floor(events_sequence_presentation[0,2]/10)*10)
    dict_sequences = {'SequenceID-%s/Position-%i'%(sequence_events[seqID],i):seqID+i  for i in range(1,7)}
    event_ids_dict = dict_scores | dict_fixation | dict_sequences

    return events_of_interest, event_ids_dict


def prepare_data_for_mne_bids_pipeline(subject,base_path = "/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/"):

    original_data_path = base_path + "/Data/raw/"

    for i in range(1,9):
        print("--- saving in bids format run %i ----"%i)
        data_path = original_data_path+subject+ '/run%i_raw.fif'%i
        raw = mne.io.read_raw_fif(data_path,allow_maxshield=True,preload=True)
        events = mne.find_events(raw,min_duration=0.01)
        # split into task and rs
        last_event_time  = (events[-1, 0] - raw.first_samp) / raw.info['sfreq']
        raw_task = raw.copy().crop(tmax=last_event_time+10)
        events, event_ids = extract_events_and_event_IDs(raw_task)

        raw_rs = raw.copy().crop(tmin=last_event_time+10)
        bids_path_rs = BIDSPath(subject='01', task='rs', run='0%i'%i,
                                 datatype='meg', root=base_path+'/Data/BIDS/')
        bids_path_task = BIDSPath(subject='01', task='reproduction', run='0%i'%i,
                                 datatype='meg', root=base_path+'/Data/BIDS/')

        write_raw_bids(raw_rs, bids_path=bids_path_rs,allow_preload=True,format='FIF',overwrite=True)
        write_raw_bids(raw_task, bids_path=bids_path_task,allow_preload=True,format='FIF',events_data=events,event_id=event_ids,overwrite=True)

        # write MEG calibration files
        cal_fname = base_path+'/Data/BIDS/system_calibration_files/sss_cal_nspn.dat'
        ct_fname = base_path+'/Data/BIDS/system_calibration_files/ct_sparse_nspn.fif'

        write_meg_calibration(calibration=cal_fname,bids_path=bids_path_rs)
        write_meg_crosstalk(fname=ct_fname,bids_path=bids_path_rs)


def prepare_data_for_mne_bids_pipeline_rs(subject, base_path="/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/"):
    original_data_path = base_path + "/Data/raw/"
    for i in range(1, 9):
        print("--- saving in bids format run %i ----" % i)
        data_path = original_data_path + subject + '/run%i_raw.fif' % i
        raw = mne.io.read_raw_fif(data_path, allow_maxshield=True, preload=True)
        events = mne.find_events(raw, min_duration=0.01)
        # split into task and rs
        last_event_time = (events[-1, 0] - raw.first_samp) / raw.info['sfreq']
        raw_rs = raw.copy().crop(tmin=last_event_time - 1)
        events_rs = mne.find_events(raw_rs, min_duration=0.01, mask_type='not_and',
                                              mask=2 ** 7 + 2 ** 8 + 2 ** 9 + 2 ** 10 + 2 ** 11 + 2 ** 12 + 2 ** 13 + 2 ** 14 + 2 ** 15 + 2 ** 16)

        event_ids = {'onset_rs':127}
        bids_path_rs = BIDSPath(subject='01', task='rs', run='0%i' % i,
                                datatype='meg', root=base_path + '/Data/BIDS/')
        write_raw_bids(raw_rs, bids_path=bids_path_rs, allow_preload=True, format='FIF',events_data=events_rs,
                       event_id=event_ids,overwrite=True)
        # write MEG calibration files
        cal_fname = base_path + '/Data/BIDS/system_calibration_files/sss_cal_nspn.dat'
        ct_fname = base_path + '/Data/BIDS/system_calibration_files/ct_sparse_nspn.fif'
        write_meg_calibration(calibration=cal_fname, bids_path=bids_path_rs)
        write_meg_crosstalk(fname=ct_fname, bids_path=bids_path_rs)


def prepare_data_for_mne_bids_pipeline_localizer(subject,base_path = "/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/"):
    original_data_path = base_path + "/Data/raw/"
    print("--- saving in bids format run localizer ----")
    data_path = original_data_path+subject+ '/localizer.fif'
    raw = mne.io.read_raw_fif(data_path,allow_maxshield=True,preload=True)
    events = mne.find_events(raw,min_duration=0.01,mask_type='not_and',mask=2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15+2**16)
    event_ids = {'Position-1':1, 'Position-2':2, 'Position-3':3, 'Position-4':4, 'Position-5':5, 'Position-6':6}

    bids_path_localizer = BIDSPath(subject='01', task='localizer',datatype='meg', root=base_path+'/Data/BIDS/')
    write_raw_bids(raw, bids_path=bids_path_localizer,allow_preload=True,format='FIF',overwrite=True, events_data=events,
                   event_id=event_ids)

    # write MEG calibration files
    cal_fname = base_path+'/Data/BIDS/system_calibration_files/sss_cal_nspn.dat'
    ct_fname = base_path+'/Data/BIDS/system_calibration_files/ct_sparse_nspn.fif'

    write_meg_calibration(calibration=cal_fname,bids_path=bids_path_localizer)
    write_meg_crosstalk(fname=ct_fname,bids_path=bids_path_localizer)




def extract_events_and_event_IDs_Antoine(raw):

    # On a deux types de données dans un même run. 10 essais de comparaison sont suivis de 40 secondes de resting state
    # Voici la timeline d'un essai : fixation, puis 500 ms après présentation de la première séquence, avec un intervalle
    # entre les 12 items de 440 ms puis une pause de 6 secondes, puis présentation de la seconde seconde puis réponse
    # du participant (TOI)
    # attention, sur toute la première partie de l'expé, il n'y avait de triggers STI008, soit 2^7 = 128
    # On va avoir des évènements :
    # croix de fixation : 10*seqID (pour 'Rep3', ça sera 30)
    # présentation des 12 items de la séquence1 : 10*seqID + position spatiale (de 1 à 6)
    # présentation des 12 items de la séquence2 : 10*seqID + position spatiale (de 1 à 6)
    # onset du resting state : 100+ 10*seqID (pour 'C2RepEmbed' ça donne 190, qui, si on n'a pas STI008, 190%128 A VERIFIER)


    sequences_codes = {'Rep2': 10, 'CRep2': 20, 'Rep3': 30, 'CRep3': 40, 'Rep4': 50, 'CRep4': 60,
                 'RepEmbed': 70, 'C1RepEmbed': 80, 'C2RepEmbed': 90}

    sequence_events = {10: 'Rep2',20: 'CRep2',30: 'Rep3',40: 'CRep3',50: 'Rep4',60: 'CRep4',70: 'RepEmbed',80: 'C1RepEmbed',90: 'C2RepEmbed'}
    # le mask permet de masker les boutons réponse (ajouter ou pas 2**7, qui correspond à STI008)
    events = mne.find_events(raw,min_duration=0.01,mask_type='not_and',mask=2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15+2**16)

    sequences = sequences_codes.keys()
    dict_fixation = {'Fixation-%s'%seq:sequences_codes[seq] for seq in sequences}
    dict_sequences = {f'Sequence-{seq}/Position-{i}':sequences_codes[seq]+i for seq in sequences for i in range(1,7)}
    dict_onset_RS = {f'Onset_RS-{seq}':100+sequences_codes[seq] for seq in sequences}
    dict_localizer = {f'Position-{i}':i for i in range(1,7)}

    # dict_sequences = {'SequenceID-%s/Position-%i'%(sequence_events[seqID],i):seqID+i  for i in range(1,7)}
    event_ids_dict =  dict_fixation | dict_sequences | dict_onset_RS | dict_localizer

    return events, event_ids_dict


def prepare_data_for_mne_bids_pipeline_Antoine(subject,base_path = "/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/",runs = ['run'+str(i) for i in range(1,10)]):

    original_data_path = base_path+"/Data/raw/"
    if len(runs)>1:
        task = 'DMTS'
    else:
        task='localizer'

    for i in range(len(runs)):
        data_path = original_data_path+subject+ '/'+runs[i]+'_raw.fif'
        raw = mne.io.read_raw_fif(data_path,allow_maxshield=True,preload=True)

        events, event_ids = extract_events_and_event_IDs_Antoine(raw)
        bids_path = mne_bids.BIDSPath(subject=subject[-2:], task=task, run=i,datatype='meg', root=base_path+'/Data/BIDS2/')

        mne_bids.write_raw_bids(raw, bids_path=bids_path,allow_preload=True,format='FIF',events_data=events,event_id=event_ids,overwrite=True)
        if task=='localizer':
            for i in range(1,10):
                bids_path = mne_bids.BIDSPath(subject=subject[-2:], task=task, run=i, datatype='meg',
                                              root=base_path + '/Data/BIDS2/')
                mne_bids.write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='FIF', events_data=events,
                                        event_id=event_ids, overwrite=True)

        # write MEG calibration files
        cal_fname = base_path+'/Data/BIDS/system_calibration_files/sss_cal_nspn.dat'
        ct_fname = base_path+'/Data/BIDS/system_calibration_files/ct_sparse_nspn.fif'
        mne_bids.write_meg_calibration(calibration=cal_fname,bids_path=bids_path)
        mne_bids.write_meg_crosstalk(fname=ct_fname,bids_path=bids_path)