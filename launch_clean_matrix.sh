#!/usr/bin/env bash
# Launch the seed-42 clean one-step matrix for our model variants with one
# PyTorch thread per worker. Omni paper values are kept as a fixed reference,
# so Omni training is opt-in here.
set -u

RUN_OMNI_BASELINE="${RUN_OMNI_BASELINE:-0}"
mkdir -p logs/clean_matrix
pids=()

run_job() {
    local name="$1"
    local variant="$2"
    local data_file="$3"
    local test_cell="$4"
    local rated="$5"
    local starts="$6"
    local -a start_args
    read -r -a start_args <<< "$starts"

    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python3 -u train_explore.py \
        --variant "$variant" \
        --data-file "$data_file" \
        --test-cell "$test_cell" \
        --rated "$rated" \
        --normalization train_minmax \
        --start-cycles "${start_args[@]}" \
        --epochs 50 --patience 10 --seed 42 \
        --rollout-horizon 4 --train-objective one_step \
        --selector-budget-weight 0 --selector-entropy-weight 0 \
        --primary-protocol observed_history \
        --torch-threads 1 --torch-interop-threads 1 \
        --out "runs/$name" \
        >"logs/clean_matrix/$name.log" 2>&1 &
    pids+=("$!")
    echo "started $name pid=${pids[-1]}"
}

# Panasonic: staircase/regeneration trajectory.
if [[ "$RUN_OMNI_BASELINE" == "1" ]]; then
    run_job clean_omni_panasonic_one_step omni Panasonic_Data.npy Cell01 3.0 "300 400 500"
fi
run_job clean_v01_panasonic_one_step adaptive Panasonic_Data.npy Cell01 3.0 "300 400 500"
run_job clean_residual_panasonic_one_step residual Panasonic_Data.npy Cell01 3.0 "300 400 500"

# TJU: dense long-cycle degradation trajectory.
if [[ "$RUN_OMNI_BASELINE" == "1" ]]; then
    run_job clean_omni_tju_one_step omni external_data/TJU_Data.npy CY25_1 2.5 "300 450 600"
fi
run_job clean_v01_tju_one_step adaptive external_data/TJU_Data.npy CY25_1 2.5 "300 450 600"
run_job clean_residual_tju_one_step residual external_data/TJU_Data.npy CY25_1 2.5 "300 450 600"

# NASA: short, sparse laboratory trajectories.
if [[ "$RUN_OMNI_BASELINE" == "1" ]]; then
    run_job clean_omni_nasa_one_step omni external_data/NASA_Data.npy B0005 2.0 "100 120"
fi
run_job clean_v01_nasa_one_step adaptive external_data/NASA_Data.npy B0005 2.0 "100 120"
run_job clean_residual_nasa_one_step residual external_data/NASA_Data.npy B0005 2.0 "100 120"

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
