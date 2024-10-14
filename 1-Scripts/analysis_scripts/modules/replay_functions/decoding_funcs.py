from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
import config
from functions import epoching_funcs, plotting_funcs
import mne
import numpy as np
import matplotlib.pyplot as plt
# ______________________________________________________________________________________________________________________

def SVM_decoder():
    """
    Builds an SVM decoder that will be able to output the distance to the hyperplane once trained on data.
    It is meant to generalize across time by construction.
    :return:
    """
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)

    return time_gen

# ______________________________________________________________________________________________________________________
def micro_avg(trial_number,n_avg):
    """
    Create a list of list of n_avg indices of trials corresponding to the trials that will be averaged to create the micro-avg
    trials.
    This function creates trial_number trials, so it means that each trial will appear in n_avg trials.
    :param trial_number:
    :param n_avg:
    :return:
    """

    counter = {str(i):n_avg for i in range(trial_number)}
    new_trials = []
    add = True
    while len(new_trials)<trial_number:
        # randomly pick n_avg from trial_number
        picked_trials = random.sample(range(trial_number), 5)
        for trial in picked_trials:
            if counter[trial] !=0:
                counter[trial] += - 1
            else:
                add = False
        if add:
            new_trials.append(picked_trials)

# ______________________________________________________________________________________________________________________

# Initialize StratifiedKFold for cross-validation

def score_per_category(decoder,X,y_true,ncv=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
    # Loop through the stratified splits
    y_true_cvs = []
    y_preds = []
    for train_index, test_index in skf.split(X, y_true):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]
        decoder.fit(X_train,y_train)
        y_preds.append(decoder.predict(X_test))
        y_true_cvs.append(y_true[test_index])

    pred_gat = np.vstack(y_preds)
    pred_diag = np.diagonal(pred_gat,axis1=1,axis2=2)

    return pred_gat, pred_diag , np.concatenate(y_true_cvs)


def train_decoder(epochs,labels,tmin=None,tmax=None,scoring=None,predict_probability=True):
    """
    train decoder on epochs from tmin to tmax, with labels 'labels'
    """
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = epochs.get_data()
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability = predict_probability))
    time_gen = GeneralizingEstimator(clf, scoring=scoring, n_jobs=8, verbose=True)
    time_gen.fit(X,labels)

    return time_gen


def test_decoder(time_gen,epochs_test,labels_test):

    projections = time_gen.decision_function(epochs_test.get_data())
    score = time_gen.score(epochs_test.get_data(),labels_test)

    return projections, score





