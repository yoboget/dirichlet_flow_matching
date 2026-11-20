#!/bin/bash
#SBATCH --job-name time_promo
#SBATCH --array=0-5
#SBATCH --error sbatch/error.e%A_%a
#SBATCH --output sbatch/out.o%A_%a
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition shared-gpu,public-gpu
#SBATCH --time 00:45:00
# Uncomment to also force the same node as the earlier timing runs:
##SBATCH --nodelist=gpu002

# Sampling time, Vector Field ('original') vs Quantile Map ('cdf_trick'), alpha_max = 8, 100 integration steps.
# One array task per (method, seed): sbatch run_time_promo.sh
# Same hardware for every task (RTX 3090, 1 CPU core). Read the results with misc/sampling_time.py.

METHODS=(original cdf_trick)
SEEDS=(0 1 2)

N_SEEDS=${#SEEDS[@]}
METHOD=${METHODS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

echo "Running task index ${SLURM_ARRAY_TASK_ID} with flow_method=${METHOD} seed=${SEED}"

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
  --lambd_ 0 \
  --eta 1 \
  --flow_method "$METHOD" \
  --alpha_max 8
