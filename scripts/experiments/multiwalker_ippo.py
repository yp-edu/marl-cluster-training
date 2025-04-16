"""
Run a single PettingZoo task with IPPO.

Run with:

```bash
uv run python -m scripts.experiments.multiwalker_ippo
```
"""

import hydra
from omegaconf import DictConfig
import warnings

from scripts.utils import validate_experiment, make_clean_folder, plot_experiment


@hydra.main(version_base=None, config_path="../../configs", config_name="exp:multiwalker_ippo")
def main(cfg: DictConfig):
    experiment = validate_experiment(cfg, ["multiwalker"], ["ippo"], ["mlp"])
    if cfg.clean_folder:
        make_clean_folder(experiment.config.save_folder)
    if cfg.train:
        experiment.run()
    if cfg.plot:
        plot_experiment(experiment, interactive=cfg.interactive)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # torchrl env warnings
    main()
