"""
Run a single experiment locally.

Run with:

```bash
uv run python -m scripts.experiments.run_local algorithm=ippo task=multiwalker/shared
```
"""

import hydra
from omegaconf import DictConfig
import warnings
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from scripts.utils import make_clean_folder, plot_experiment, load_experiment
from scripts.setup import setup_custom_tasks


@setup_custom_tasks
@hydra.main(version_base=None, config_path="../../configs", config_name="exp:run_local")
def main(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices
    algorithm_name = hydra_choices.algorithm
    task_name = hydra_choices.task
    logger.info(f"Algorithm: {algorithm_name}, Task: {task_name}")

    experiment = load_experiment(cfg)

    if cfg.clean_folder:
        make_clean_folder(experiment.config.save_folder)
    if cfg.train:
        experiment.run()
    if cfg.plot:
        plot_experiment(experiment, interactive=cfg.interactive)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # torchrl env warnings
    main()
