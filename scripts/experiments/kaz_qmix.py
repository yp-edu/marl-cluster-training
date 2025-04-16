"""
Run a single PettingZoo task with QMIX.

Run with:

```bash
uv run python -m scripts.experiments.kaz_qmix
```
"""

import hydra
from omegaconf import DictConfig
import warnings


from scripts.utils import make_clean_folder, validate_experiment, plot_experiment


@hydra.main(version_base=None, config_path="../../configs", config_name="exp:kaz_qmix")
def main(cfg: DictConfig):
    experiment = validate_experiment(cfg, ["kaz"], ["qmix"], ["mlp"])
    if cfg.clean_folder:
        make_clean_folder(experiment.config.save_folder)
    if cfg.train:
        experiment.run()
    if cfg.plot:
        plot_experiment(experiment, interactive=cfg.interactive)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # torchrl env warnings
    main()
