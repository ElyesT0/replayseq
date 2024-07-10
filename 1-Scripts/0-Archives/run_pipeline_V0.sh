#!/bin/bash

# Run the first command
mne_bids_pipeline --config config_items.py 

# Check if the first command was successful
if [ $? -eq 0 ]; then
    # Run the second command
    mne_bids_pipeline --config config_sequences.py 
else
    echo "The first command failed, the second command will not be executed."
fi
