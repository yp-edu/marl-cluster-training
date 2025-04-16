"""
Run a single Kaz task.

Run with:

```bash
uv run python -m scripts.kaz
```
"""

from benchmarl.algorithms import IppoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.cnn import CnnConfig

from scripts.utils import make_clean_folder
from src.core.task import KazTask


def main():
    experiment_config = ExperimentConfig.get_from_yaml()
    task = KazTask.KAZ.get_from_yaml("conf/task/kaz/kaz.yaml")
    algorithm_config = IppoConfig.get_from_yaml()
    model_config = CnnConfig.get_from_yaml()
    critic_model_config = CnnConfig.get_from_yaml()

    experiment_config.evaluation_interval = experiment_config.off_policy_collected_frames_per_batch
    experiment_config.max_n_iters = 5
    experiment_config.save_folder = "results/pettingzoo/"
    make_clean_folder(experiment_config.save_folder)

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()
