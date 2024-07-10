#!/bin/bash

# Function to run a command, check if it was successful, and log the output
run_command() {
    local cmd="$1"
    local description="$2"
    local logfile="$3"
    
    echo "Running: $description" | tee -a "$logfile"
    $cmd 2>&1 | tee -a "$logfile"
    if [ $? -ne 0 ]; then
        echo "$description failed, proceeding to the next one." | tee -a "$logfile"
    else
        echo "$description succeeded." | tee -a "$logfile"
    fi
}

# Define the commands
command1="mne_bids_pipeline --config config_sequences.py"
command2="mne_bids_pipeline --config config_items.py"
command3="mne_bids_pipeline --config config_resting_stage.py"

# Define the log file
logfile="pipeline_log.txt"

# Clear the log file at the start
> "$logfile"

# Run the commands
run_command "$command1" "First command" "$logfile"
run_command "$command2" "Second command" "$logfile"
run_command "$command3" "Third command" "$logfile"
