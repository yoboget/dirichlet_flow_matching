#!/bin/bash
#SBATCH --job-name time_fly
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
# One array task per (method, seed): sbatch run_time_flybrain.sh
# Same hardware for every task (RTX 3090, 1 CPU core). Read the results with misc/sampling_time.py.

METHODS=(original cdf_trick)
SEEDS=(0 1 2)

N_SEEDS=${#SEEDS[@]}
METHOD=${METHODS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

echo "Running task index ${SLURM_ARRAY_TASK_ID} with flow_method=${METHOD} seed=${SEED}"

srun apptainer exec --nv \
  --env LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH \
  seq.sif python -m train_dna --run_name FlyBrain --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 0 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop --validate --validate_on_test --ckpt workdir/DNA_valOnTrain_epoch12Eval_uncondFID_FIDearlyS_probAddGuidance3_2024-01-07_16-23-01/epoch=1329-step=436240.ckpt --eta 1 --lambd_ 0 --flow_method "$METHOD" --seed "$SEED" --alpha_max 8
