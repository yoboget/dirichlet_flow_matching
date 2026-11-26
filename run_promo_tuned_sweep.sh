#!/bin/bash
#SBATCH --job-name promo_tuned_sweep
#SBATCH --array=0-14
#SBATCH --error sbatch/error.e%A_%a
#SBATCH --output sbatch/out.o%A_%a
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition shared-gpu,public-gpu
#SBATCH --time 00:45:00

# Sweep of the 'tuned' sampler on Promoter along the diagonal lambda = eta (as DIAGONAL=true in
# ModeFlowGraph/run_tuned_sweep.sh): for each ETA value, one run with --eta ETA --lambd_ ETA.
#   eta    : 0 = conditional quantile map, 1 = total resampling
#   lambd_ : 0 = sample one class, > 0 = Dirichlet(lambda * p) mixing weights, -1 = +inf (weighted average)
# One array task per (eta, seed): 5 x 3 = 15 tasks. Submit from the repo root:
#   mkdir -p sbatch && sbatch run_promo_tuned_sweep.sh
# If you change ETA or SEEDS, set --array=0-(len(ETA) * len(SEEDS) - 1).
# Same checkpoint, alpha_max, NFE and hardware as the Promoter eta sweep with lambda = 0 (tuned_8_* runs).
# W&B group: tuned_8.0_lambda<eta>_eta<eta>  (fetch with misc/run_groups.py: fetch_group_runs(..., lambd_=eta, eta=eta))

ETA=(0.05 0.25 0.5 0.75 1)
SEEDS=(0 1 2)

N_ETA=${#ETA[@]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / N_ETA ))]}
E=${ETA[$(( SLURM_ARRAY_TASK_ID % N_ETA ))]}

echo "Running task index ${SLURM_ARRAY_TASK_ID} with eta=lambda=${E} seed=${SEED}"

srun apptainer exec --nv \
  --env LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH \
  seq.sif python -m train_promo \
  --run_name train_dirichlet_fm \
  --seed "$SEED" \
  --batch_size 128 \
  --wandb \
  --num_workers 4 \
  --check_val_every_n_epoch 5 \
  --num_integration_steps 100 \
  --validate \
  --validate_on_test \
  --ckpt workdir/promo_diri_2024-01-31_10-49-42/epoch=14-step=10380-Copy1.ckpt \
  --flow_method tuned \
  --eta "$E" \
  --lambd_ "$E" \
  --alpha_max 8
