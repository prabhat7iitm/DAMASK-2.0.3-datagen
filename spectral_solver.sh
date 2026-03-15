# Set the start time
module unload anaconda3
start_time=$(date +%s)
export DAMASK_NUM_THREADS=8 ; echo $DAMASK_NUM_THREADS
DAMASK_spectral --geometry *.geom --load *.load

# Set the end time
end_time=$(date +%s)

# Calculate the runtime
runtime=$((end_time - start_time))

echo "Script runtime: $runtime seconds"
