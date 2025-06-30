"""
Run a benchmark for the multiwalker task.

Run with:

```bash
uv run -m scripts.run_benchmark \
    algorithm@algs.a1=ippo \
    +algorithm@algs.a2=mappo \
    task@tasks.t1=pettingzoo/multiwalker \
    +task@tasks.t2=multiwalker/shared \
    experiment=debug_no_log \
    plot_environment=pettingzoo \
    plot=true \
    interactive=true
```
"""

import hydra
from omegaconf import DictConfig
import warnings
from hydra.core.hydra_config import HydraConfig

from scripts.utils import load_benchmark, plot_experiments, make_clean_folder, get_experiment_json_file
from scripts.setup import setup_custom_tasks


@setup_custom_tasks
@hydra.main(version_base=None, config_path="../configs", config_name="run_benchmark")
def main(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices
    benchmark = load_benchmark(cfg)
    if cfg.clean_folder:
        make_clean_folder(benchmark.experiment_config.save_folder)
    if cfg.train:
        experiments_json_files = []
        for experiment in benchmark.get_experiments():
            experiments_json_files.append(get_experiment_json_file(experiment))
            experiment.run()

    if cfg.plot:
        if cfg.plot_environment is None:
            raise ValueError("plot_environment must be set if plot is true")
        plot_experiments(
            experiments_json_files,
            cfg.plot_environment,
            [[hydra_choices[f"algorithm@algs.{alg_alias}"] for alg_alias in benchmark.algorithm_configs.keys()]],
            interactive=cfg.interactive,
            save_folder=benchmark.experiment_config.save_folder,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # torchrl env warnings
    main()
