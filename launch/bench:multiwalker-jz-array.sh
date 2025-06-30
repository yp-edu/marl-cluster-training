#!/bin/bash

#SBATCH --job-name=bench:multiwalker-jz
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1200
#SBATCH --mail-type=ALL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --account=nwq@v100
#SBATCH --array=0-11

module purge
uv run --no-sync -m scripts.benchmarks.multiwalker \
    run_specific_expirment=$SLURM_ARRAY_TASK_ID \
    algorithm@algs.a1=ippo \
    +algorithm@algs.a2=mappo \
    task@tasks.t1=pettingzoo/multiwalker \
    +task@tasks.t2=multiwalker/shared \
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
