import mne
import pandas as pd
from dipy.viz.projections import matplotlib

from modules.replay_functions import replay_funcs, epoching_funcs, decoding_funcs, plotting_funcs
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from decoding_package.decoding_functions import *
from numpy import matrix
import tdlm
import config
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
from decoding_package.plotting_funcs import *
import matplotlib.pyplot as plt

# %matplotlib qt


subjects = ['01','02','03','05','07','08']
subjects = ['01','03','04']

# ----------------------------------------------------------------------------------------------------------------------
# --------------    DETERMINE IF DECODER IS WORKING FINE AND OPTIMIZE IT     -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
subject = '01'

for subject in subjects:

    epochs = epoching_funcs.create_training_epochs_7categories(subject)
    epochs.apply_baseline(baseline=(0,0.05))

    epochs.pick_types(meg=True)
    PCA_model = UnsupervisedSpatialFilter(PCA(70), average=False)
    PCA_data = PCA_model.fit_transform(epochs.get_data())
    epochs._data = PCA_data
    epochs_7cat = epochs.copy()
    epochs_6positions = epochs_7cat[epochs_7cat.events[:,2]<10]

    conditions = {'7conds_baseline_full_window':epochs_7cat,'6conds_baseline_full_window':epochs_6positions}

    for labels, epo in conditions.items():
        # 1 - run simple model - 6 positions
        modelgat = GeneralizingEstimator(decoding_model(Bagging=False))
        One_score_simple_gat = cross_val_multiscore(modelgat,X = epo.get_data(),y=epo.events[:,2])
        fig = pretty_gat(np.mean(One_score_simple_gat,axis=0),times=epo.times,chance=1/len(np.unique(epo.events[:,2])))
        fig.savefig(config.figure_path+'/decoding/sub-'+subject+'_simple_model-'+labels+'.svg')

# ----------------------------------------------------------------------------------------------------------------------
# --------------    TRAIN A DECODER ON THE 7 CATEGORIES (6 POSITIONS + NULL) -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
epochs = epoching_funcs.create_training_epochs_7categories(subject)
epochs = epochs[epochs.events[:, 2] < 10]
epochs.apply_baseline(baseline=(0, 0.05))

epochs_train = epochs.copy().pick_types(meg=True)
# PCA 70 components, that will help speed up
pca = UnsupervisedSpatialFilter(PCA(70), average=False)
pca_data = pca.fit_transform(epochs_train.get_data())
epochs_train._data = pca_data

# train decoder
trained_decoder = decoding_funcs.train_decoder(epochs_train,labels=epochs.events[:,2])

# Fit the same PCA to the epochs test
epochs_test = mne.read_epochs(config.derivatives_path+'/sequence/sub-'+subject+'/meg/sub-'+subject+'_task-reproduction_epo.fif')
epochs_test.apply_baseline(baseline=(-0.05,0))
epochs_test._data = pca.transform(epochs_test.pick_types(meg=True).get_data())

# predict probabilities of each of the 7 states during the full sequence
reactivation_probability = trained_decoder.predict_proba(epochs_test.get_data())

# astuce pour avoir un dataframe qui contient les metadata pour chaque epoques mais aussi les probas prédites !
pred_probs_df = epoching_funcs.load_behavioral_file(subject)
pred_probs_df['predicted_probas'] = [reactivation_probability[i] for i in range(len(pred_probs_df))]
pred_probs_df.to_pickle(config.results_path+"/reactivation/sub-"+subject+"_predicted_probabilities.pkl")

# ----------------------------------------------------------------------------------------------------------------------
# --------------  First "Sanity check" : are presented position more reactivated during the break  ---------------------
# ----------------------------------------------------------------------------------------------------------------------

reactivation_presented_vs_notpresented = replay_funcs.compute_reactivation(epoching_funcs.load_behavioral_file(subject), reactivation_probability, train_times = epochs_train.times, test_times = epochs_test.times, plot=True)
plotting_funcs.pretty_gat(np.mean(reactivation_presented_vs_notpresented, axis=0), epochs_train.times, epochs_test.times, chance=0)

reactivation_df = epoching_funcs.load_behavioral_file(subject)
reactivation_df['predicted_probas'] = [reactivation_presented_vs_notpresented[i] for i in range(len(reactivation_df))]
reactivation_df.to_pickle(config.results_path+"/reactivation/sub-"+subject+"_reactivation.pkl")

# ----------------------------------------------------------------------------------------------------------------------
# -------------- COMPUTE EMPIRICAL TRANSITION MATRIX WITH SEVERAL VALUES OF LAGS ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Let's imagine that the time of maximal decoding performance corresponds to the data point number 10
# Then, it means that you just want to look at the reactivation probabilities for this specific training time

# 1 - X_k are the reactivation_probabilities (predictions from the localizer decoder) as a function of time for state k (state for us : 1, 2, ... 6 positions spations + 7 = null)
# 2 - Empirical transition matrix Beta : 2D, index i,j : how much the reactivation_probability of state j at t + delta t is related to the
# the reactivation_probability of state i at t

tmax = 37 # environ 150 ms après stim onset
predicted_p = reactivation_probability[:, tmax, :, :6]

lags = np.arange(1,100,1)
n_epochs = reactivation_probability.shape[0]
n_lags = len(lags)
betas_epochs = np.zeros((n_epochs,n_lags,6,6))
for n_trial in range(n_epochs):
    pred_probs = predicted_p[n_trial,:,:]
    betas = []
    for ii, delta in enumerate(lags):
        beta_delta = replay_funcs.compute_beta(pred_probs,delta)
        betas_epochs[n_trial,ii,:,:] = beta_delta

# --------- PREDICTOR MATRICES : Forward, Backward, autoregression, and controls made of the order of the presented items -------
num_states = 6
metadata = epoching_funcs.load_behavioral_file(subject)
forwards = np.zeros((n_epochs, num_states,num_states))
backwards = np.zeros((n_epochs, num_states,num_states))
autocorrelation = np.zeros((n_epochs, num_states,num_states))
constant = np.ones((n_epochs, num_states,num_states))
# add the shuffled and the autocorrelation

for n_trial in range(n_epochs):
    forward_matrix, backward_matrix = replay_funcs.calculate_transition_matrices(metadata.iloc[n_trial]['PresentedSequence'])
    forwards[n_trial,:]= (forward_matrix)
    backwards[n_trial,:]= (backward_matrix)
    autocorrelation[n_trial,:]= (np.identity(6))

n_trials = 15*2*9
seq_forward = []
seq_backward = []
for k in range(n_trials):
    print("----- trial %i -----"%k)
    preds = predicted_p[0,:,:]
    tf = forwards[0,:,:] # transition matrix
    sequenceness_fwd, sequenceness_bkw = tdlm.compute_1step(preds, tf)
    seq_forward.append(sequenceness_fwd)
    seq_backward.append(sequenceness_bkw)

# plot results
tdlm.plotting.plot_sequenceness(np.mean(seq_forward,axis=0), np.mean(seq_backward,axis=0))
plt.show()

tlag = 0

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Fit the model to the training data

empirical_matrix = np.asarray([ matrix.flatten(np.asarray([matrix.flatten(betas_epochs[i,ii,:,:]) for i in range(betas_epochs.shape[0])])) for ii in range(n_lags)])
Predictors = {'forward':matrix.flatten(forwards), 'backward':matrix.flatten(backwards), 'autocorrelation':matrix.flatten(autocorrelation), 'constant':matrix.flatten(constant)}



coeffs = []
for tlag in range(n_lags):
    Dependant_variable = pd.DataFrame({'Empirical_matrix':empirical_matrix[tlag]})
    model.fit(Predictors[['forward']], Dependant_variable['Empirical_matrix'])
    coeffs.append(model.coef_)



betas = np.asarray(coeffs)
plt.plot()
plt.show()
# ----------- and now, linear regression of the empirical transition matrix as a function of different transition matrices ------------------

import numpy as np
from sklearn.linear_model import LinearRegression
X = np.asarray([np.matrix.flatten(forwards),np.matrix.flatten(backwards),np.matrix.flatten(autocorrelation),np.matrix.flatten(constant)])


regression_coefficients = []
for ii,tlag in enumerate(lags):
    y = np.asarray([np.matrix.flatten(betas_epochs[i,ii,:,:]) for i in range(betas_epochs.shape[0])])
    y = np.matrix.flatten(y)
    reg = LinearRegression()
    reg.fit(X.T, y)
    regression_coefficients.append(reg.coef_)

regression_coefficients = np.asarray(regression_coefficients)

sequenceness = regression_coefficients[:,0]-regression_coefficients[:,1]

plt.plot(regression_coefficients[:,3])
plt.show()
