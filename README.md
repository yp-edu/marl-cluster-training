# MARL cluster training

This repository contains the code for the MARL cluster training blog post [->](https://yp-edu.github.io/projects/marl-cluster-training).

## Usage

Install the dependencies with `uv`:

```bash
uv sync
```

Run an experiment:

```bash
uv run -m scripts.run_experiment \
    algorithm=ippo \
    task=multiwalker/shared \
    experiment=debug_no_log \
    model=layers/mlp
```

That's mostly it! See the following resources for more details:

- Python package manager: [uv](https://docs.astral.sh/uv/)
- Main framework: [BenchMARL](https://benchmarl.readthedocs.io/en/latest/)
- Configs: [Hydra](https://hydra.cc/)
- Cluster launcher: [SLURM](https://slurm.schedmd.com/documentation.html)
