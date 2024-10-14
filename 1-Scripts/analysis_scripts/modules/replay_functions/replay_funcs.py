# In this module are all the functions related to the sequenceness analysis we want to run (adapted from
# Human Replay Spontaneously Reorganizes Experience, Cell 2019, Liu et al.)

from __future__ import division
import os.path as op
import pickle
import random

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from functions import plotting_funcs, epoching_funcs, decoding_funcs
import pandas as pd
import config
import mne

# def train_position_decoder(subject,tmin=None,tmax=None,method='mean'):
#
#     dec = decoding_funcs.SVM_decoder()
#     epo = epoching_funcs.epoch_localizer_data(subject)
#     epo.crop(tmin=tmin,tmax=tmax)
#     epo.equalize_event_counts()
#     if method=='mean':
#         train_data = np.mean(epo.get_data(),axis=-1)
#         dec.fit(train_data[:,:,np.newaxis],y=epo.events[:,2])
#     else:
#         dec.fit(epo.get_data(), y=epo.events[:, 2])
#     np.save(config.results_path+'/Decoding/Spatial_code/'+subject+'_tmin-'+str(np.round(tmin*1000,0))+'_tmax-'+str(np.round(tmax*1000,0))+'_method-'+method,dec)
#     return dec

def train_position_decoder(subject,tmin=None,tmax=None,method='mean'):

    print('This is a substitute of the function that should be based on the data from the localizer part')

    dec = decoding_funcs.SVM_decoder()
    epo = mne.read_epochs(config.data_path+subject+'/meg_item/'+subject+'_task-reproduction_epo.fif')
    epo = [epo['Position-%i' % i] for i in range(1, 7)]
    mne.epochs.equalize_epoch_counts(epo)
    labels = []
    for ii in range(len(epo)):
        labels.append([ii+1] * len(epo[ii]))
    labels = np.hstack(labels)

    epo = mne.concatenate_epochs(epo)
    epo.pick_types(meg='mag')
    epo.crop(tmin=tmin,tmax=tmax)
    if method=='mean':
        train_data = np.mean(epo.get_data(),axis=-1)
        dec.fit(train_data[:,:,np.newaxis],y=labels)
    elif method == 'alltimepoints':
        dec.fit(epo.get_data(), y=labels)
    else:
        dec.fit(epo.get_data(), y=labels)
    np.save(config.results_path+'/Decoding/Spatial_code/'+subject+'_tmin-'+str(np.round(tmin*1000,0))+'_tmax-'+str(np.round(tmax*1000,0))+'_method-'+method,dec)
    return dec

def score_position_decoder(subject,tmin=None,tmax=None):

    print('This is a substitute of the function that should be based on the data from the localizer part')

    dec = decoding_funcs.SVM_decoder()
    epo = mne.read_epochs(config.data_path+subject+'/meg_item/'+subject+'_task-reproduction_epo.fif')
    epo = [epo['Position-%i' % i] for i in range(1, 7)]
    mne.epochs.equalize_epoch_counts(epo)
    labels = []
    for ii in range(len(epo)):
        labels.append([ii+1] * len(epo[ii]))
    labels = np.hstack(labels)
    epo = mne.concatenate_epochs(epo)
    epo.pick_types(meg='mag')
    epo.crop(tmin=tmin,tmax=tmax)
    score = mne.decoding.cross_val_multiscore(dec,epo.get_data(),labels)
    np.save(config.results_path+'/Decoding/Spatial_code/'+subject+'_tmin-'+str(np.round(tmin*1000,0))+'_tmax-'+str(np.round(tmax*1000,0))+'SCORE',score)
    return dec


# ---------------         --------------       -------------
def compute_reactivation(metadata, predicted_probas, train_times,test_times, plot=True):
    """
    Compute the reactivation timeline based on presented and non-presented positions.

    Parameters:
    -----------
    metadata : DataFrame
        DataFrame containing the 'PresentedPositions' for each trial.
    predicted_probas : ndarray
        3D numpy array of predicted probabilities with shape (n_trials, n_timepoints, n_positions).
    times : ndarray
        1D numpy array of time points corresponding to the predictions.
    plot : bool, optional
        If True, the reactivation will be plotted using `plotting_funcs.pretty_decod`. Default is True.

    Returns:
    --------
    reactivation : ndarray
        2D numpy array of reactivation values with shape (n_trials, n_timepoints).
    """
    reactivation = np.zeros((len(metadata),len(train_times),len(test_times)))
    presented_sequences = metadata['PresentedSequence'].values

    for trial in range(len(metadata)):
        print(f"Processing trial {trial + 1}/{len(metadata)}")
        presented_trial = np.unique(presented_sequences[trial])
        not_presented = list(set(range(6)) - set(presented_trial))  # assuming positions are [0,1,2,3,4,5]
        # check that this is the right dimensionality - I'm not sure since the decoder has actually several decoding points
        predicted_trial = predicted_probas[trial, :]
        reactivated_trial = np.mean(predicted_trial[:,:, presented_trial], axis=-1) - \
                            np.mean(predicted_trial[:,:, not_presented], axis=-1)
        reactivation[trial,:] = reactivated_trial

    if plot:
        plotting_funcs.pretty_gat(np.mean(reactivation,axis=0), train_times,test_times, chance=0)

    return reactivation



def compute_beta(X, delta_t):
    """
    Computes the beta coefficient matrix as per the provided equation.

    Parameters:
    X (numpy.ndarray): Matrix of size (n_states, times) representing the time series.
    delta_t (int): The time delay to apply to X to create X(Delta_t).

    Returns:
    numpy.ndarray: The computed beta coefficients.
    """
    # Ensure delta_t is a valid index
    if delta_t < 0 or delta_t >= X.shape[0]:
        raise ValueError("Delta_t must be between 0 and the number of time steps in X minus 1")

    # Create the delayed time series X(Delta_t)
    X_delta_t = np.roll(X, shift=-delta_t, axis=1)

    # Remove the last delta_t columns to avoid boundary effects
    X_delta_t = X_delta_t[:-delta_t,:]
    X_trimmed = X[:-delta_t,:]

    # Compute beta = (X^T X)^-1 X^T X(Delta_t)
    XT_X = np.matmul(X_trimmed.T, X_trimmed)
    XT_X_inv = np.linalg.inv(XT_X)
    XT_X_delta_t = np.matmul(X_trimmed.T, X_delta_t)

    beta = np.dot(XT_X_inv, XT_X_delta_t)

    return beta

# ______________________________________________________________________________________________________________________
def calculate_transition_matrices(sequence, num_states=6):
    # Initialize transition matrices
    forward_matrix = np.zeros((num_states, num_states))
    backward_matrix = np.zeros((num_states, num_states))

    # Calculate forward transition counts
    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        forward_matrix[current_state][next_state] += 1

    # Normalize forward transition counts to get probabilities
    for i in range(num_states):
        row_sum = np.sum(forward_matrix[i])
        if row_sum > 0:
            forward_matrix[i] /= row_sum

    # Calculate backward transition counts
    for i in range(1, len(sequence)):
        current_state = sequence[i]
        previous_state = sequence[i - 1]
        backward_matrix[current_state][previous_state] += 1

    # Normalize backward transition counts to get probabilities
    for i in range(num_states):
        row_sum = np.sum(backward_matrix[i])
        if row_sum > 0:
            backward_matrix[i] /= row_sum

    return forward_matrix, backward_matrix



def test_position_decoder(subject,tmin,tmax,method='mean',test_time_window = 'maintenance'):

    decoder_path = config.results_path+'Decoding/Spatial_code/'+subject+'_tmin-'+str(np.round(tmin*1000,0))+'_tmax-'+str(np.round(tmax*1000,0))+'_method-'+method+'.npy'
    if op.exists(decoder_path):
        dec = np.load(decoder_path,allow_pickle=True).item()
    else:
        dec = train_position_decoder(subject, tmin=tmin, tmax=tmax, method=method)

    if test_time_window=='maintenance':
        path_data_test = config.data_path + "/" + subject + "/meg_full_trial/"+subject+"_task-reproduction_epo.fif"
        epo_test = mne.read_epochs(path_data_test)
        epo_test.crop(tmin=config.fixation_duration+config.SOA*config.nitems,tmax=config.fixation_duration+config.SOA*config.nitems+config.maintenance_duration)
        epo_test.pick_types(meg='mag')
        epo_test.decimate(10)
    elif test_time_window == 'presentation':
        path_data_test = config.data_path + "/" + subject + "/meg_full_trial/"+subject+"_task-reproduction_epo.fif"
        epo_test = mne.read_epochs(path_data_test)
        epo_test.crop(tmin=0,tmax=config.SOA*config.nitems)
    elif test_time_window=='all':
        path_data_test = config.data_path + "/" + subject + "/meg_full_trial/"+subject+"_task-reproduction_epo.fif"
        epo_test = mne.read_epochs(path_data_test)
        epo_test.crop(tmin=0,tmax=config.fixation_duration+config.SOA*config.nitems+config.maintenance_duration)
        epo_test.pick_types(meg='mag')
        epo_test.decimate(10)

    test_data = epo_test.get_data()
    Y = dec.predict_proba(test_data)
    Y = np.squeeze(Y)

    return Y, epo_test.times, epo_test.metadata

# ______________________________________________________________________________________________________________________
def sequence_delta(proba_time_series,expected_sequence, delta_npoints):
    """
    This function sums over the starting points the state probabilities shifted by delta_t for the sequence of states
    defined by expected_sequence
    :param proba_time_series: n_times X n_states
    :param expected_sequence: which sequence of states do we want to test
    :param delta_npoints: number of time points assumed between the states
    :return: integral of the sequential activations
    """
    somme = 0
    k = 0
    while k <  proba_time_series.shape[0]-delta_npoints*len(expected_sequence):
        for ii, state in enumerate(expected_sequence):
            somme+= proba_time_series[int(k+ii*delta_npoints),state]
        k+=1

    return somme/k


def compute_sequence_replay_trial(Predicted_Probas_trial, deltas, replayed_sequence_trial):
    seq_replay = []
    for delta_npoints in deltas:
        seq_replay_delta = sequence_delta(Predicted_Probas_trial, replayed_sequence_trial, delta_npoints)
        seq_replay.append(seq_replay_delta)
    seq_replay = np.asarray(seq_replay)

    return seq_replay, deltas

# ______________________________________________________________________________________________________________________
def replay_presented_sequences_in_maintenance(subject,tmin=0.140,tmax=0.160,deltas = range(1,50),build_H0=True):

    Predicted_Probas, _, metadata = test_position_decoder(subject, tmin, tmax, method='mean', test_time_window='maintenance')

    metadata = epoching_funcs.read_behavioral_data(config.behavioral_results+subject+'/exp_01.xpd')
    presented_sequences = metadata['PresentedPositions'].values

    seq_replay = []
    seq_replay_H0 = []
    for trial in range(Predicted_Probas.shape[0]):
        print("================ Running for trial %i ============"%trial)
        seq_replay_trial, deltas = compute_sequence_replay_trial(Predicted_Probas[trial], deltas, presented_sequences[trial])
        if build_H0:

            seq_replay_H0_trial = []
            k=0
            while k<1000:
                shuffle_trial_seq = presented_sequences[trial]
                random.shuffle(shuffle_trial_seq)
                seq_replay_trial_shuff, deltas = compute_sequence_replay_trial(Predicted_Probas[trial], deltas,shuffle_trial_seq)
                seq_replay_H0_trial.append(seq_replay_trial_shuff)
                k+=1
        seq_replay_H0.append(np.asarray(seq_replay_H0_trial))
        seq_replay.append(seq_replay_trial)
    seq_replay = np.asarray(seq_replay)
    seq_replay_H0 = np.asarray(seq_replay_H0)

    dictionnary = {'sequence_replay':seq_replay.T,'time_lags':deltas,'metadata':metadata,'seq_replay_H0':seq_replay_H0}
    np.save(config.results_path+'/Replay/Spatial_code/'+subject+'_timewindow-maintenance.npy',dictionnary)

    return seq_replay.T, deltas, metadata

# ______________________________________________________________________________________________________________________
def sanity_check_replay_presented_sequences(subject,tmin=0.140,tmax=0.160,deltas = [int(i) for i in np.linspace(300,500,20)]):

    Predicted_Probas, _ = test_position_decoder(subject, tmin, tmax, method='mean', test_time_window='presentation')

    metadata = epoching_funcs.read_behavioral_data(config.behavioral_results+subject+'/exp_01.xpd')
    presented_sequences = metadata['PresentedPositions'].values
    seq_delta = []
    for delta_npoints in deltas:
        print("---- running for delta %i ----"%int(delta_npoints))
        seq_replay = []
        for trial in range(Predicted_Probas.shape[0]):
            seq_replay_delta = sequence_delta(Predicted_Probas[trial], presented_sequences[trial], delta_npoints)
            seq_replay.append(seq_replay_delta)
        seq_delta.append(seq_replay)
    seq_delta = np.asarray(seq_delta)

    dictionnary = {'sequence_replay':seq_delta.T,'time_lags':deltas,'metadata':metadata}
    np.save(config.results_path+'/Replay/Spatial_code/'+subject+'_timewindow-presentation.npy',dictionnary)

    return seq_delta.T, deltas, metadata


def plot_replay_sequenceID(subject,test_time_window='maintenance',sequences=None):

    # seq_list = ['RepEmbed','Rep3','Rep2','Rep4','C1RepEmbed','C2RepEmbed','CRep2','CRep3','CRep4']

    if test_time_window=='maintenance':
        data = np.load(config.results_path+'/Replay/Spatial_code/'+subject+'_timewindow-maintenance.npy',allow_pickle=True).item()
    elif test_time_window=='presentation':
        data = np.load(config.results_path+'/Replay/Spatial_code/'+subject+'_timewindow-presentation.npy',allow_pickle=True).item()


    seq_delta_replay = data['sequence_replay'].T
    time_lags = data['time_lags']
    metadata = data['metadata']
    H0 = False
    if 'seq_replay_H0' in data.keys():
        H0 = True
        seq_replay_H0 = data['seq_replay_H0']

    if sequences is not None:
        fig, axs = plt.subplots(len(sequences))
        for ii, SeqID in enumerate(sequences):
            indices = np.where(metadata["SequenceID"].values==SeqID)[0]
            plt.axes(axs[ii])
            plot_H0_95interval(seq_replay_H0[indices,:,:],axs[ii],time_lags)
            axs[ii].set_title(SeqID)
            axs[ii].plot([i / 1000 for i in time_lags],np.mean(seq_delta_replay[indices,:],axis=0))

    else:
        ax = plotting_funcs.pretty_decod(seq_delta_replay, times=[i / 1000 for i in time_lags],
                                    chance=np.mean(seq_delta_replay))


def plot_H0_95interval(H0_matrix,ax,time_lags):

    for n in range(H0_matrix.shape[1]):
        for k in range(H0_matrix.shape[0]):
            plt.plot([i / 1000 for i in time_lags], H0_matrix[k,n,:],color="lightgrey",alpha=0.05)
    twosigma = [np.mean(H0_matrix[:,:,i]) + 2*np.std(H0_matrix[:,:,i]) for i in range(H0_matrix.shape[2])]
    ax.plot([i / 1000 for i in time_lags], twosigma, color="red")
#   # add extra thickness








def sliding_window_predict_proba(decode_dict,epochs_replay,tstart=0):
    """
    This function tests the decoder at different time-points.
    The output is a nbins X nepochs X n_labels
    :param decoder:
    :param epochs:
    :param y:
    :return:
    """

    decoder = decode_dict['decoder']
    decoder_mean = decode_dict['mode']=='mean'
    tmax = epochs_replay.tmax
    if decoder_mean:
        delta_t=0.001
    else:
        delta_t = np.round(decode_dict['tmax']-decode_dict['tmin'],4)
    nbins = int(np.floor((tmax-tstart)/delta_t))
    tmin_array = [np.round(tstart+i*delta_t,4) for i in range(nbins)]
    data_size = decode_dict['data_size']

    total_predictions = []
    for tmin in tmin_array:
        print(tmin)
        epo_data = epochs_replay.copy().crop(tmin=tmin,tmax=tmin+delta_t)
        preds_prob = decoder.predict_proba(np.squeeze(epo_data.get_data()[:,:,:data_size]))
        total_predictions.append(preds_prob)

    total_predictions = np.asarray(total_predictions)
    total_predictions = np.transpose(total_predictions,[1,2,0])

    predict_proba_dict = epochs_replay.metadata
    total_predictions_meta = [total_predictions[K,:,:] for K in range(len(total_predictions))]
    predict_proba_dict['predicted_probas'] =total_predictions_meta
    predict_proba_dict['tmin'] = decode_dict['tmin']
    predict_proba_dict['tmax'] = decode_dict['tmax']
    predict_proba_dict['decim']= decode_dict['decim']

    return predict_proba_dict

# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
#                                     TRANSITION MATRICES AND THEIR RANDOMIZATION
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

def transition_matrix(sequence,transition_matrix_type = 'spatial_locations',plot_transition_matrix = False):

    transition_mat = np.zeros((6,6))
    for k in range(5):
        transition_mat[sequence[k],sequence[k+1]] = 1

    if transition_matrix_type == 'spatial_locations':
        labels = ['pos1','pos2','pos3','pos4','pos5','pos6']

    if plot_transition_matrix:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.imshow(transition_mat)

    return transition_mat


# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
#                                           GLM MEASURES AND SEQUENCENESS
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

def sequenceness_all_trials(predict_proba_dict,lags_values = np.arange(0,0.5,0.001),transition_matrix_type='spatial_locations'):

    """
    :param predict_proba_dict:
    :param lags_values: list of the different lags for which we want to compute the sequenceness
    :param transition_matrix_type:
    :return:
    """

    n_trials = len(predict_proba_dict)

    # ===== for each trial (i.e. each sequence) extract the equations to solve ========
    sequenceness_mat = np.zeros((n_trials,len(lags_values)))

    for trial in range(n_trials):
        print('==== running the sequenceness analysis for trial number %i ===='%trial)
        SequenceDisplayed = predict_proba_dict['SequenceDisplayed'].iloc[trial]
        SequenceDisplayed = [x-1 for x in SequenceDisplayed]
        print(trial)

        for kk, lag in enumerate(lags_values):
            betas = glm_one_seq_one_lag(predict_proba_dict,trial,lag)

            SequenceDisplayed_rev = SequenceDisplayed.copy()
            SequenceDisplayed_rev.reverse()

            transition_mat_forward = transition_matrix(SequenceDisplayed,transition_matrix_type = transition_matrix_type)
            transition_mat_backward = transition_matrix(SequenceDisplayed_rev,transition_matrix_type = transition_matrix_type)
            sequenceness_mat[trial,kk] = sequenceness_computation(betas,transition_mat_forward,transition_mat_backward)


    return sequenceness_mat



def glm_one_seq_one_lag(predict_proba_dict,trial_number,lag):

    decim = predict_proba_dict['decim'][0]
    predicted_probas = predict_proba_dict['predicted_probas'].iloc[trial_number]
    n_states, n_times = predicted_probas.shape
    if decim is None:
        lag_data_points = int(lag*1000)
    else:
        lag_data_points = int(lag * 1000 / decim)
    # ===== one regression per final state ======

    BETAS = np.zeros((n_states,n_states))
    y = np.zeros((n_states,(n_times-lag_data_points)))
    X = np.zeros((n_states,(n_times-lag_data_points),n_states))

    for nstate in range(n_states):
        for pp, t in enumerate(range(lag_data_points,n_times)):
            y[nstate,pp] = predicted_probas[nstate,t]
            X[nstate,pp,:] = predicted_probas[:,t-lag_data_points]

        linear_model = LinearRegression()
        y_reg = np.squeeze(y[nstate, :])
        X_reg = X[nstate,:,:]
        reg = linear_model.fit(X_reg, y_reg)
        linear_model.score(X_reg, y_reg)
        BETAS[nstate,:] = reg.coef_

    return BETAS


def sequenceness_computation(betas,transition_mat_forward,transition_mat_backward):

    betas_reshaped = np.reshape(betas, (1,np.product(betas.shape)))
    transition_mat_forward_reshaped = np.reshape(transition_mat_forward, (1,np.product(transition_mat_forward.shape)))
    transition_mat_backward_reshaped = np.reshape(transition_mat_backward, (1,np.product(transition_mat_backward.shape)))

    return np.inner(betas_reshaped,transition_mat_forward_reshaped)[0][0]-np.inner(betas_reshaped,transition_mat_backward_reshaped)[0][0]


# ========= full wrap up function ============

def compute_sequenceness(subject,lags_values = np.arange(0,0.5,0.001),transition_matrix_type='spatial_locations'):

    replay_subject_dir = op.join(config.decoding_dir, subject, 'replay')
    with open(replay_subject_dir+op.sep+'replay_results.pkl','rb') as fid:
        dict_results = pickle.load(fid)
    predict_proba_dict = dict_results['metadata_predict_proba_dict']
    sequenceness = sequenceness_all_trials(predict_proba_dict, lags_values=lags_values,
                            transition_matrix_type=transition_matrix_type)
    save_name = replay_subject_dir + 'sequenceness_' + transition_matrix_type + '.npy'

    np.save(save_name,sequenceness)



def load_sequenceness_matrix(subject,transition_matrix_type='spatial_locations'):


    replay_subject_dir = op.join(config.decoding_dir, subject, 'replay')
    fname = replay_subject_dir + 'sequenceness_' + transition_matrix_type + '.npy'

    return np.load(fname)


def compute_reactivation_score(subject,load_name = '/replay/replay_results',save_name='/position_reactivation.pkl',mode = 'mean'):

    # ====== load all the data that contains the predicted things ============
    save_subject_dir = op.join(config.decoding_dir, subject)
    load_path = save_subject_dir+ load_name+'.pkl'
    with open(load_path, 'rb') as fid:
        prediction_dict = pickle.load(fid)

    predictions = prediction_dict['metadata_predict_proba_dict']

    if mode == 'mean':
        preds = predictions['predicted_probas']
    else:
        preds = predictions['y_preds']

    sequence = predictions['SequenceDisplayed']

    reactivation_score = []
    for k in range(len(predictions)):
        print("====== running the loop for trial %i ======="%k)
        # we loop through the lines of the dictionnary
        positions_disp = np.unique(sequence[k])
        positions_not_disp = list(set(range(1,7))- set(positions_disp) )
        if mode == 'mean':
            # Careful, are the positions coded from 1 to 6 or from 0 to 5 ?
            react_disp = np.mean(preds[k][list(positions_disp-1),:],axis= 0)
            react_not_disp = np.mean(preds[k][list(np.array(positions_not_disp)-1),:],axis=0)
            reactivation_score.append(react_disp - react_not_disp)
        if mode != 'mean':
            react_disp = np.isin(preds[k],positions_disp)
            reactivation_score.append(react_disp)


    # ======= now for each trial output a score of how much the presented locations (i.e. np.unique(sequence1)) are more
    # reactivated than the others. Add this to the initial dictionnary and save it ================

    predictions['reactivation_score'] = reactivation_score
    print('== The reactivation score was appended to the results dictionnary ==')

    predictions = pd.DataFrame.from_dict(predictions)
    predictions['Subject'] = [subject]*len(predictions)

    save_subject_dir = op.join(config.decoding_dir, subject,'reactivation')
    utils.create_folder(save_subject_dir)
    with open(save_subject_dir+save_name,'wb') as fid:
        pickle.dump(predictions,fid)
    np.save(save_subject_dir+'/position_reactivation.npy',predictions)

    return predictions






def load_and_plot_reactivation(save_name='/reactivation/position_reactivation',mode = 'mean',plot_figure=True):

    reactivation = []

    for subject in config.subjects_list:

        save_subject_dir = op.join(config.decoding_dir, subject)
        load_path = save_subject_dir+ save_name
        with open(load_path, 'rb') as fid:
            reactivation_dict = pickle.load(fid)

        reactivation.append(np.vstack(reactivation_dict['reactivation_score'].values))

    reactivation = np.asarray(reactivation)

    # ========= plot the results ===========
    if plot_figure:
        plt.close('all')
        times = np.round(np.linspace(0,2.4,reactivation.shape[2]),3)
        if len(reactivation.shape)==2:
            fig = GFP_funcs.plot_mean_with_sem(np.mean(reactivation,axis=1),times=times)
            plt.ylabel('Mean activation presented - mean activation not presented')
        else:
            fig = plotting_funcs.pretty_gat(np.mean(reactivation,axis=0),times = times,chance=0,)
        plt.title('Reactivation')
        plt.xlabel('Time[s]')
        plt.axhline(y=0)
        # ======== savefig path ============
        path_fig = op.join(config.fig_path,'Reactivation/')
        fig.savefig(path_fig+'reactivation.png',dpi=300)


    return reactivation