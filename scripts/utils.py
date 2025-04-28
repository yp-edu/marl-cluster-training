import os
import shutil
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

from benchmarl.experiment import Experiment
from benchmarl.benchmark import Benchmark
from benchmarl.eval_results import load_and_merge_json_dicts, Plotting
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
    load_experiment_from_hydra,
)


def make_clean_folder(folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def load_experiment(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices

    task_name = hydra_choices.task
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)

    if not Path(experiment_config.save_folder).exists():
        os.makedirs(experiment_config.save_folder)

    return load_experiment_from_hydra(cfg, task_name=task_name)


def load_benchmark(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices

    algorithm_configs = [load_algorithm_config_from_hydra(alg_config) for alg_config in cfg.algs.values()]
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    tasks = [
        load_task_config_from_hydra(task_config, hydra_choices[f"task@tasks.{task_alias}"])
        for task_alias, task_config in cfg.tasks.items()
    ]
    model_config = load_model_config_from_hydra(cfg.model)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)

    if not Path(experiment_config.save_folder).exists():
        os.makedirs(experiment_config.save_folder)

    return Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        model_config=model_config,
        critic_model_config=critic_model_config,
        experiment_config=experiment_config,
        seeds=cfg.seeds,
    )


def get_experiment_json_file(experiment: Experiment):
    return str(Path(experiment.folder_name) / Path(experiment.name + ".json"))


def plot_experiment(experiment: Experiment, interactive: bool = True):
    raw_dict = load_and_merge_json_dicts([get_experiment_json_file(experiment)])
    processed_data = Plotting.process_data(raw_dict)
    (
        environment_comparison_matrix,
        _,
    ) = Plotting.create_matrices(processed_data, env_name=experiment.environment_name)

    # Plotting
    perf_profiles_fig = Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    sample_eff_fig = Plotting.task_sample_efficiency_curves(
        processed_data=processed_data, env=experiment.environment_name, task=experiment.task.name
    )
    if interactive:
        plt.show()
    else:
        perf_profiles_fig.figure.savefig(Path(experiment.folder_name) / Path("performance_profiles.png"))
        sample_eff_fig.figure.savefig(Path(experiment.folder_name) / Path("sample_efficiency.png"))


def plot_experiments(
    experiments_json_files: list[str],
    env_name: str,
    algorithms_to_compare: list[list[str]],
    interactive: bool = True,
    save_folder: Optional[str] = None,
):
    raw_dict = load_and_merge_json_dicts(experiments_json_files)
    processed_data = Plotting.process_data(raw_dict)
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name=env_name)
    perf_profiles_fig = Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.aggregate_scores(environment_comparison_matrix=environment_comparison_matrix)
    sample_eff_fig, _, _ = Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix
    )
    prob_of_imp_fig = Plotting.probability_of_improvement(
        environment_comparison_matrix,
        algorithms_to_compare=algorithms_to_compare,
    )
    if interactive:
        plt.show()
    else:
        if save_folder is None:
            raise ValueError("save_folder must be provided if interactive is False")
        perf_profiles_fig.figure.savefig(Path(save_folder) / Path("performance_profiles.png"))
        sample_eff_fig.figure.savefig(Path(save_folder) / Path("sample_efficiency.png"))
        prob_of_imp_fig.figure.savefig(Path(save_folder) / Path("probability_of_improvement.png"))
