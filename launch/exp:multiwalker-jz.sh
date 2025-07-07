#!/bin/bash

#SBATCH --job-name=exp:multiwalker-jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-16g
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --account=nwq@v100

N_FRAMES=${1:-10_000_000}
PREVIOUS_JOBID=${2:-null}
PREVIOUS_N_FRAMES=${3:-10000000}
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

echo "N_FRAMES: $N_FRAMES"
echo "PREVIOUS_JOBID: $PREVIOUS_JOBID"
echo "PREVIOUS_N_FRAMES: $PREVIOUS_N_FRAMES"
echo "RESTORE_FOLDER: $RESTORE_FOLDER"
echo "RESTORE_FILE: $RESTORE_FILE"

module purge
uv run --no-sync -m scripts.run_experiment \
    train=true \
    plot=false \
    interactive=false \
    seed=2 \
    \
    algorithm=mappo \
    task=pettingzoo/multiwalker \
    model=layers/mlp \
    model.activation_class=torch.nn.Tanh \
    \
    +experiment.wandb_extra_kwargs.offline=true \
    experiment.train_device=cpu \
    experiment.sampling_device=cpu \
    experiment.buffer_device=cpu \
    \
    experiment.collect_with_grad=true \
    experiment.parallel_collection=true \
    \
    experiment.max_n_frames=$N_FRAMES \
    experiment.lr=0.0005 \
    \
    experiment.on_policy_collected_frames_per_batch=100_000 \
    experiment.on_policy_n_envs_per_worker=10 \
    experiment.on_policy_n_minibatch_iters=5 \
    experiment.on_policy_minibatch_size=80_000 \
    \
    experiment.evaluation_interval=1_000_000 \
    experiment.evaluation_episodes=5 \
    experiment.restore_file=$RESTORE_FILE \
    experiment.checkpoint_at_end=true \
    experiment.checkpoint_interval=1_000_000
