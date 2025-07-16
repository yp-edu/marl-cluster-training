#!/bin/bash

#SBATCH --job-name=exp:multiwalker-jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=10
##SBATCH --partition=gpu_p2l
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --account=nwq@v100

N_FRAMES=${1:-100_000_000}
PREVIOUS_JOBID=${2:-null}
PREVIOUS_N_FRAMES=${3:-100000000}
RESTORE_FILE="null"

if [[ "$PREVIOUS_JOBID" != "null" ]]; then
    OUTFILE="results/slurm/exp:multiwalker-jz-${PREVIOUS_JOBID}.err"
    if [[ -f "$OUTFILE" ]]; then
        RESTORE_FOLDER=$(awk -F': ' '/Folder[[:space:]]*name:/ {print $2}' "$OUTFILE")
        if [[ -n "$RESTORE_FOLDER" ]]; then
            RESTORE_FILE="$RESTORE_FOLDER/checkpoints/checkpoint_${PREVIOUS_N_FRAMES}.pt"
        fi
    fi
fi

FRAMES_PER_BATCH=20_000
N_ENVS_PER_WORKER=10
N_MINIBATCH_ITERS=20
MINIBATCH_SIZE=10_000
EVALUATION_INTERVAL=10_000_000
MEMORY_SIZE=10_000_000
LR=0.0003

echo "--------------------------------"
echo "N_FRAMES: $N_FRAMES"
echo "FRAMES_PER_BATCH: $FRAMES_PER_BATCH"
echo "N_ENVS_PER_WORKER: $N_ENVS_PER_WORKER"
echo "N_MINIBATCH_ITERS: $N_MINIBATCH_ITERS"
echo "MINIBATCH_SIZE: $MINIBATCH_SIZE"
echo "MEMORY_SIZE: $MEMORY_SIZE"
echo "LR: $LR"
echo "--------------------------------"
echo "EVALUATION_INTERVAL: $EVALUATION_INTERVAL"
echo "--------------------------------"
echo "PREVIOUS_JOBID: $PREVIOUS_JOBID"
echo "PREVIOUS_N_FRAMES: $PREVIOUS_N_FRAMES"
echo "RESTORE_FOLDER: $RESTORE_FOLDER"
echo "RESTORE_FILE: $RESTORE_FILE"

module purge
uv run --no-sync -m scripts.run_experiment \
    train=true \
    plot=false \
    interactive=false \
    seed=42 \
    \
    algorithm=mappo \
    task=pettingzoo/multiwalker \
    model=layers/mlp \
    model.activation_class=torch.nn.ReLU \
    model.num_cells="[256, 256]" \
    \
    +experiment.wandb_extra_kwargs.offline=true \
    experiment.train_device=cuda \
    experiment.sampling_device=cpu \
    experiment.buffer_device=cuda \
    \
    experiment.collect_with_grad=false \
    experiment.parallel_collection=true \
    \
    experiment.max_n_frames=$N_FRAMES \
    experiment.lr=$LR \
    \
    experiment.on_policy_collected_frames_per_batch=$FRAMES_PER_BATCH \
    experiment.on_policy_n_envs_per_worker=$N_ENVS_PER_WORKER \
    experiment.on_policy_n_minibatch_iters=$N_MINIBATCH_ITERS \
    experiment.on_policy_minibatch_size=$MINIBATCH_SIZE \
    \
    experiment.off_policy_collected_frames_per_batch=$FRAMES_PER_BATCH \
    experiment.off_policy_n_envs_per_worker=$N_ENVS_PER_WORKER \
    experiment.off_policy_n_optimizer_steps=$N_MINIBATCH_ITERS \
    experiment.off_policy_train_batch_size=$MINIBATCH_SIZE \
    experiment.off_policy_memory_size=$MEMORY_SIZE \
    \
    experiment.evaluation_interval=$EVALUATION_INTERVAL \
    experiment.evaluation_episodes=5 \
    experiment.restore_file=$RESTORE_FILE \
    experiment.checkpoint_at_end=true \
    experiment.checkpoint_interval=$EVALUATION_INTERVAL
