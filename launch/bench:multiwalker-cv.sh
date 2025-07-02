#!/bin/bash

#SBATCH --job-name=bench:multiwalker-cv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

module purge
uv run --no-sync -m scripts.run_benchmark \
    algorithm@algs.a1=ippo \
    +algorithm@algs.a2=mappo \
    task@tasks.t1=pettingzoo/multiwalker \
    +task@tasks.t2=multiwalker/shared \
    train=true \
    plot=false \
    interactive=false \
    experiment=gpu_offline \
    \
    experiment.collect_with_grad=true \
    experiment.parallel_collection=true \
    \
    experiment.max_n_frames=500_000 \
    experiment.lr=0.00005 \
    \
    experiment.on_policy_collected_frames_per_batch=5_000 \
    experiment.on_policy_n_minibatch_iters=10 \
    experiment.on_policy_n_envs_per_worker=10 \
    experiment.evaluation_interval=50_000
