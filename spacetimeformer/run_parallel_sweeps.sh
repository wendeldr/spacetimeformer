#!/bin/bash

# Default number of parallel runs
DEFAULT_NUM_RUNS=1

# Check if an argument is provided, else use default
if [ $# -eq 0 ]
then
    NUM_RUNS=$DEFAULT_NUM_RUNS
else
    NUM_RUNS=$1
fi

# Function to run the Python script
run_sweep () {
    python wandb_sweep.py &
}

echo "Starting $NUM_RUNS parallel wandb sweep runs..."

# Loop to start multiple runs in the background
for (( i=0; i<$NUM_RUNS; i++ ))
do
    run_sweep
    echo "Run $i started."
done

# Wait for all background processes to finish
wait
echo "All wandb sweep runs have completed."

