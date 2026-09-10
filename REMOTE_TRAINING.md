# Autodl training recipe

The `remote-training` branch contains the current observed-history protocol and
the recursive rollout diagnostic.  Run the following on the GPU instance.

```bash
git clone -b remote-training https://github.com/HiddenKismet/HeteroTIEFormer.git
cd HeteroTIEFormer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch einops numpy pandas requests
python tools/prepare_batteryarchive_calce.py
```

On the 32-core CPU, launch both variants concurrently and pin 16 cores to each
process.  The new `--torch-threads` option prevents PyTorch from silently
falling back to one thread.  Both use the same split, measured-capacity
windows, seed, and stopping rule:

```bash
mkdir -p runs/remote_logs

nohup taskset -c 0-15 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
python -u train_explore.py \
  --variant omni \
  --data-file external_data/BatteryArchive_CALCE_CX2.npy \
  --test-cell 'CALCE_CX2-16_prism_LCO_25C_0-100_0.5/0.5C_a' \
  --rated 1.35 --normalization train_minmax \
  --start-cycles 300 400 500 \
  --epochs 50 --patience 10 --seed 42 \
  --d-model 16 --batch-size 1024 --rollout-horizon 4 \
  --torch-threads 16 --torch-interop-threads 1 \
  --train-objective scheduled_sampling \
  --primary-protocol observed_history \
  --regional-restore legacy \
  --out runs/ba_calce_a_omni_remote \
  > runs/remote_logs/ba_calce_a_omni_remote.log 2>&1 &

nohup taskset -c 16-31 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
python -u train_explore.py \
  --variant adaptive \
  --data-file external_data/BatteryArchive_CALCE_CX2.npy \
  --test-cell 'CALCE_CX2-16_prism_LCO_25C_0-100_0.5/0.5C_a' \
  --rated 1.35 --normalization train_minmax \
  --start-cycles 300 400 500 \
  --epochs 50 --patience 10 --seed 42 \
  --d-model 16 --batch-size 1024 --rollout-horizon 4 \
  --torch-threads 16 --torch-interop-threads 1 \
  --train-objective scheduled_sampling \
  --primary-protocol observed_history \
  --regional-restore aligned \
  --out runs/ba_calce_a_v01_remote \
  > runs/remote_logs/ba_calce_a_v01_remote.log 2>&1 &
```

Progress is available with `tail -f runs/remote_logs/*.log`.  The final
paper-style metrics are written to each run's `metrics.json`; `primary` is the
observed-history result and `recursive` is the secondary free-running
diagnostic.
