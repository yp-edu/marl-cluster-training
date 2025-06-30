#!/bin/bash

#SBATCH --job-name=exp:multiwalker-jz
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1200
#SBATCH --mail-type=ALL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --account=nwq@v100

module purge
uv run --no-sync -m scripts.run_experiment \
    algorithm=ippo \
    task=pettingzoo/multiwalker \
    train=true \
    plot=false \
    interactive=false \
    experiment=gpu_offline \
    experiment.sampling_device=cpu \
    experiment.train_device=cuda \
    experiment.buffer_device=cpu \
    experiment.on_policy_collected_frames_per_batch=24000 \
    experiment.off_policy_collected_frames_per_batch=24000 \
    experiment.evaluation_interval=480000
