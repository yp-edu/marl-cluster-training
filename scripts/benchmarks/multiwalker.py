"""
Run a benchmark for the multiwalker task.

Run with:

```bash
uv run python -m scripts.benchmarks.multiwalker
```
"""

import hydra
from omegaconf import DictConfig
import warnings

from scripts.utils import load_benchmark, plot_experiments, make_clean_folder, get_experiment_json_file


@hydra.main(version_base=None, config_path="../../configs", config_name="bench:multiwalker")
def main(cfg: DictConfig):
    benchmark = load_benchmark(cfg)
    if cfg.clean_folder:
        make_clean_folder(benchmark.experiment_config.save_folder)
    if cfg.train:
        experiments_json_files = []
        for experiment in benchmark.get_experiments():
            experiments_json_files.append(get_experiment_json_file(experiment))
            experiment.run()

    if cfg.plot:
        plot_experiments(
            experiments_json_files,
            "pettingzoo",
            "multiwalker",
            [["ippo", "mappo"]],
            interactive=cfg.interactive,
            save_folder=benchmark.experiment_config.save_folder,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # torchrl env warnings
    main()
