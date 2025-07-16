#!/bin/bash

#SBATCH --job-name=bench:multiwalker-jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --account=nwq@v100
#SBATCH --array=0-11

module purge
uv run --no-sync -m scripts.run_benchmark \
    algorithm@algs.a1=ippo \
    +algorithm@algs.a2=mappo \
    task@tasks.t1=pettingzoo/multiwalker \
    +task@tasks.t2=multiwalker/shared \
    run_specific_expirment=$SLURM_ARRAY_TASK_ID \
    \
    train=true \
    plot=false \
    interactive=false \
    \
    +experiment.wandb_extra_kwargs.offline=true \
    experiment.train_device=cuda \
    experiment.sampling_device=cpu \
    experiment.buffer_device=cuda \
    \
    experiment.collect_with_grad=false \
    experiment.parallel_collection=true \
    \
    experiment.max_n_frames=10_000_000 \
    experiment.lr=0.00008 \
    \
    experiment.on_policy_collected_frames_per_batch=40_000 \
    experiment.on_policy_n_envs_per_worker=10 \
    experiment.on_policy_n_minibatch_iters=10 \
    experiment.on_policy_minibatch_size=20_000 \
    \
    experiment.evaluation_interval=1_000_000 \
    experiment.evaluation_episodes=5
