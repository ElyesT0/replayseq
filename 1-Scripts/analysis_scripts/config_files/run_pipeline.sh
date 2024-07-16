#!/bin/bash

# Function to run a command, check if it was successful, and log the output
run_command() {
    local cmd="$1"
    local description="$2"
    local logfile="$3"
    
    echo "Running: $description" | tee -a "$logfile"
    eval "$cmd" 2>&1 | tee -a "$logfile"
    if [ $? -ne 0 ]; then
        echo "$description failed, proceeding to the next one." | tee -a "$logfile"
    else
        echo "$description succeeded." | tee -a "$logfile"
    fi
}

# Get the directory of the script
script_dir=$(dirname "$0")

# Define the configurations and steps
configs=("config_sequences.py" "config_items.py" "config_resting_stage.py")
steps=("01" "02" "03" "04" "05" "06a1" "06a2" "07")

# Define the log file
logfile="pipeline_log.txt"

# Clear the log file at the start
> "$logfile"

# Run the commands
total_commands=$(( ${#configs[@]} * ${#steps[@]} ))
count=0

for config in "${configs[@]}"; do
    for step in "${steps[@]}"; do
        count=$((count + 1))
        cmd="mne_bids_pipeline --config $script_dir/$config --steps=preprocessing/$step"
        description="Running $config with step preprocessing/$step"
        run_command "$cmd" "$description" "$logfile"
        echo -e "\n=======================================" | tee -a "$logfile"
        echo -e "Command $count of $total_commands completed." | tee -a "$logfile"
        echo -e "$count commands ran. $((total_commands-count)) left to run." | tee -a "$logfile"
        echo -e "=======================================\n" | tee -a "$logfile"
    done
done
