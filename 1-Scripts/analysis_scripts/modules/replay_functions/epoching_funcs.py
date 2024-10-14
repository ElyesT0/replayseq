import pandas as pd
import config
import mne
import numpy as np
import pandas as pd
import warnings
from glob import glob


def from_seq_to_seqID(sequence):

    seq = sequence.replace('[','')
    seq = seq.replace(']','')
    seq = seq.split(',')
    seq = [int(i) for i in seq]
    new_seq = np.asarray([0]*12)
    l = 0
    A = seq[l]
    inds_A = np.where(np.asarray(seq) == A)[0]
    new_seq[inds_A] = l
    pres_pos = [A]
    for ii in range(1,12):
        if seq[ii] not in pres_pos:
            l += 1
            A = seq[ii]
            inds_A = np.where(np.asarray(seq) == A)[0]
            new_seq[inds_A] = l
            pres_pos.append(A)

    SEQS = {'[0 1 0 1 0 1 0 1 0 1 0 1]':'Rep2','[0 1 1 1 1 0 0 1 0 0 0 1]':'CRep2',
    '[0 1 2 0 1 2 0 1 2 0 1 2]':'Rep3', '[0 1 2 0 2 1 1 2 0 1 0 2]':'CRep3',
    '[0 1 2 3 0 1 2 3 0 1 2 3]':'Rep4','[0 1 2 3 2 1 3 0 0 3 1 2]':'CRep4',
    '[0 0 1 1 2 2 0 0 1 1 2 2]':'RepEmbed', '[0 0 1 1 2 2 0 0 2 2 1 1]':'C1RepEmbed',
    '[0 1 2 0 2 1 0 1 2 0 2 1]':'C2RepEmbed'}

    return (SEQS[str(new_seq)])

def load_behavioral_file(subject, start_keyword='subject_id'):
    """
    Extracts and processes a pandas DataFrame from a CSV file by dynamically finding the start of relevant data,
    and converts specific string representations of lists into actual lists.

    Parameters:
    - file_path: str, the path to the CSV file.
    - start_keyword: str, the keyword that indicates the start of relevant data (default is 'subject_id').

    Returns:
    - df: pandas DataFrame, containing the extracted and processed data.
    """
    start_line = None
    if subject == '01' or subject == '02':
        file_path = glob(config.behavior_raw_path + "/sub-" + subject + "/*.csv")
        start_line = 0
    else:
        file_path = glob(config.behavior_raw_path + "/sub-" + subject + "/*.xpd")

    file_path = file_path[0]
    # Read the file into a list of lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line number where the relevant data starts
    if start_line !=0:
        for i, line in enumerate(lines):
            if start_keyword in line:
                start_line = i
                break

    # Load the CSV starting from the identified line
    df = pd.read_csv(file_path, skiprows=start_line, index_col=False)

    if subject == '01' or subject == '02':
        df.drop(columns=['participant_number'])
        df['subject_id'] = int(subject)
        df['sequenceName'] = [from_seq_to_seqID(seq) for seq in df['PresentedSequence'].values]
    else:
        columns_of_interest = [
            'subject_id', 'block', 'sequenceName', 'trial',
            'PresentedSequence', 'ProducedSequence', 'RTs', 'Performance'
        ]
        df = df[columns_of_interest]

    # Select only the columns of interest

    # Convert string representations of lists to actual lists for relevant columns
    list_columns = ['PresentedSequence', 'ProducedSequence', 'RTs']
    for col in list_columns:
        df[col] = df[col].apply(eval)

    return df


def expand_dataframe_with_position(df):
    """
    Expands the dataframe by duplicating each row 12 times, corresponding to each item in the PresentedSequence,
    and adds a column 'PresentedPosition' that contains each of the 12 items.

    Parameters:
    - df: pandas DataFrame, the original dataframe.

    Returns:
    - expanded_df: pandas DataFrame, the expanded dataframe with 'PresentedPosition' column added.
    """
    # Initialize an empty list to store the expanded rows
    expanded_rows = []

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        # Get the PresentedSequence for the current row
        presented_sequence = row['PresentedSequence']

        # Check if the length of PresentedSequence is 12 (as expected)
        if len(presented_sequence) != 12:
            raise ValueError("PresentedSequence does not have exactly 12 items.")

        # Create 12 new rows, one for each item in PresentedSequence
        for i, item in enumerate(presented_sequence):
            # Copy the original row and add the PresentedPosition
            new_row = row.copy()
            new_row['PresentedPosition'] = item
            expanded_rows.append(new_row)

    # Convert the list of expanded rows into a new DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df



#_______________________________________________________________________________________________________________________
def extract_first_occurrences(lst):
    first_occurrences = {}
    seen = set()

    for index, value in enumerate(lst):
        if value not in seen:
            seen.add(value)
            first_occurrences[index] = value

    return first_occurrences

#_______________________________________________________________________________________________________________________
def extract_epochs_first_presentation(subject):
    """
    This function extracts the epochs corresponding to the first presentation of spatial items for a given subject, 
    specifically those whose positions could not be anticipated.

    Parameters:
    subject (str): The subject identifier (used to construct the file path and filter based on subject ID)

    Returns:
    mne.Epochs: The epochs of spatial items that were presented first in each sequence and could not be anticipated.
    """
    import warnings

    # Define the path to the preprocessed MEG epochs file based on the subject ID
    path = config.derivatives_path + '/items/sub-' + subject + '/meg/sub-' + subject + '_task-reproduction_epo.fif'
    
    # Issue a warning that this function works only for epochs before rejection
    warnings.warn("Careful, this function only works for epochs before rejection", UserWarning)
    
    # Load the epochs with preload enabled (data loaded into memory)
    epochs = mne.read_epochs(path, preload=True)
    
    # Special case for subjects '01' and '02': remove epochs with event code 1
    if subject == '01' or subject == '02':
        epochs = epochs[epochs.events[:, 2] != 1]  # Filter out epochs where event code is 1
    
    # Load the subject's behavioral metadata and expand it with spatial position information
    metadata = load_behavioral_file(subject)
    epochs.metadata = expand_dataframe_with_position(metadata)  # Attach expanded metadata to the epochs
    
    # Extract the "PresentedSequence" from metadata
    presented_sequences = metadata["PresentedSequence"].values
    
    # Initialize a list to hold the indices of the first occurrences of each item in the sequence
    indices = []
    
    # For each sequence in "PresentedSequence", find the first occurrence of each item
    for k, seq in enumerate(presented_sequences):
        first_occurrences = extract_first_occurrences(seq)  # Get first occurrences in the sequence
        indices.append([i + 12 * k for i in list(first_occurrences.keys())])  # Adjust indices for sequence position
    
    # Concatenate all indices into a single array
    indices = np.concatenate(indices)
    
    # Return the epochs corresponding to the first occurrences in each sequence
    return epochs[indices]


#_______________________________________________________________________________________________________________________
def extract_epoch_null_category(subject):
    """
    This function loads MEG fixation epochs for a given subject, crops them into three 500 ms epochs, 
    and returns the concatenated spatial epochs that could not be anticipated.

    Parameters:
    subject (str): The subject identifier (used to construct file path)

    Returns:
    mne.EpochsArray: A new EpochsArray containing three cropped epochs (each 500 ms long) concatenated vertically.
    """
    
    # Define the path to the preprocessed MEG epochs file based on the subject ID
    path = config.derivatives_path + 'fixation/sub-'+subject+'/meg/sub-'+subject+'_task-reproduction_epo.fif'
    
    # Load the MEG epochs from the file with preload enabled (data loaded into memory)
    epochs = mne.read_epochs(path, preload=True)
    
    # Concatenate the data from three cropped time intervals:
    # 1. From -1 to -0.5 seconds
    # 2. From -0.5 to 0 seconds
    # 3. From 0 to 0.5 seconds
    data = np.vstack([
        epochs.copy().crop(tmin=-1, tmax=-0.5).get_data(),  # First 500 ms epoch
        epochs.copy().crop(tmin=-0.5, tmax=0).get_data(),   # Second 500 ms epoch
        epochs.copy().crop(tmin=0, tmax=0.5).get_data()     # Third 500 ms epoch
    ])
    
    # Create a new EpochsArray object using the concatenated data and the original epochs' info
    epochs = mne.EpochsArray(data=data, info=epochs.copy().crop(tmin=0, tmax=0.5).info)

    # Return the new EpochsArray
    return epochs


# _______________________________________________________________________________________________________________________
def create_training_epochs_7categories(subject):
    """
    This function creates training epochs for a given subject by combining spatial item epochs and null epochs.
    It adjusts for potential differences in sampling frequency, assigns event codes, and returns a concatenated 
    set of epochs with equalized event counts across categories.

    Parameters:
    subject (str): The subject identifier (used to extract epochs)

    Returns:
    mne.Epochs: A concatenated MNE Epochs object containing epochs of 6 spatial categories and 1 null category.
    """
    
    # Extract the null category epochs and set event code to 99 for all null events
    epochs_null = extract_epoch_null_category(subject)
    epochs_null.events[:, 2] = 99  # Set all event codes in null epochs to 99
    
    # Extract the item presentation epochs and set event codes based on 'PresentedPosition' metadata
    epochs_items = extract_epochs_first_presentation(subject)
    epochs_items.events[:, 2] = epochs_items.metadata['PresentedPosition'].values  # Assign event codes from metadata
    
    # Handle potential differences in sampling frequency between the two datasets
    sfreq_i, sfreq_n = epochs_items.info['sfreq'], epochs_null.info['sfreq']  # Get sampling frequencies
    min_sfreq = np.min([sfreq_i, sfreq_n])  # Find the minimum sampling frequency
    if sfreq_i != min_sfreq:
        epochs_items.resample(sfreq=min_sfreq)  # Resample epochs_items if its sampling rate is higher
    elif sfreq_n != min_sfreq:
        epochs_null.resample(sfreq=min_sfreq)  # Resample epochs_null if its sampling rate is higher

    # Crop epochs_items to match the time range of epochs_null
    epochs_items.crop(epochs_null.tmin, epochs_null.tmax)
    
    # Remove metadata from epochs_items to prevent issues during concatenation
    epochs_items.metadata = None

    # Create MNE EpochsArray objects with labeled event IDs for both item and null epochs
    epochs_i = mne.EpochsArray(
        epochs_items, 
        epochs_items.info, 
        events=epochs_items.events, 
        tmin=0.0, 
        event_id={
            'Position_1': 0, 'Position_2': 1, 'Position_3': 2, 'Position_4': 3, 'Position_5': 4, 'Position_6': 5
        }
    )
    epochs_n = mne.EpochsArray(
        epochs_null, 
        epochs_null.info, 
        events=epochs_null.events, 
        tmin=0.0, 
        event_id={'null': 99}
    )

    # Concatenate item and null epochs into a single MNE Epochs object
    epochs = mne.concatenate_epochs([epochs_i, epochs_n])

    # Equalize the number of events across all categories
    epochs.equalize_event_counts()

    # Return the concatenated and balanced epochs
    return epochs
