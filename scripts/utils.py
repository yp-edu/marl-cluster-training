import os
import shutil
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

from benchmarl.algorithms import IppoConfig, MappoConfig, QmixConfig, AlgorithmConfig
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig, ModelConfig
from benchmarl.benchmark import Benchmark
from benchmarl.eval_results import load_and_merge_json_dicts, Plotting


def make_clean_folder(folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def update_config_(config: DictConfig, update_dict: DictConfig):
    for key, value in update_dict.items():
        setattr(config, key, value)


def validate_task(task_name: str, allowed_tasks: list[str], task_cfg: DictConfig) -> PettingZooTask:
    if task_name not in allowed_tasks:
        raise ValueError(f"Task {task_name} not among allowed tasks: {allowed_tasks}")
    if task_name == "multiwalker":
        task = PettingZooTask.MULTIWALKER.get_from_yaml()
    else:
        raise NotImplementedError(f"Task {task_name} not implemented")
    task.update_config(task_cfg)
    return task


def validate_algorithm(
    algorithm_name: str, allowed_algorithms: list[str], algorithm_cfg: DictConfig
) -> AlgorithmConfig:
    if algorithm_name not in allowed_algorithms:
        raise ValueError(f"Algorithm {algorithm_name} not among allowed algorithms: {allowed_algorithms}")
    if algorithm_name == "ippo":
        algorithm_config = IppoConfig.get_from_yaml()
    elif algorithm_name == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
    elif algorithm_name == "qmix":
        algorithm_config = QmixConfig.get_from_yaml()
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not implemented")
    update_config_(algorithm_config, algorithm_cfg)
    return algorithm_config


def validate_model(model_name: str, allowed_models: list[str], model_cfg: DictConfig) -> ModelConfig:
    if model_name not in allowed_models:
        raise ValueError(f"Model {model_name} not among allowed models: {allowed_models}")
    if model_name == "mlp":
        model_config = MlpConfig.get_from_yaml()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    update_config_(model_config, model_cfg)
    return model_config


def validate_experiment(
    cfg: DictConfig,
    allowed_tasks: list[str],
    allowed_algorithms: list[str],
    allowed_models: list[str],
) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task.split("/")[0]
    algorithm_name = hydra_choices.algorithm.split("/")[0]
    model_name = hydra_choices.model.split("/")[0]
    critic_model_name = hydra_choices["model@critic_model"].split("/")[0]

    experiment_config = ExperimentConfig.get_from_yaml()
    update_config_(experiment_config, cfg.experiment)

    task = validate_task(task_name, allowed_tasks, cfg.task)
    algorithm_config = validate_algorithm(algorithm_name, allowed_algorithms, cfg.algorithm)
    model_config = validate_model(model_name, allowed_models, cfg.model)
    critic_model_config = validate_model(critic_model_name, allowed_models, cfg.critic_model)

    if not Path(experiment_config.save_folder).exists():
        os.makedirs(experiment_config.save_folder)

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        config=experiment_config,
        seed=cfg.seed,
    )
    return experiment


def validate_benchmark(
    cfg: DictConfig, allowed_tasks: list[str], allowed_algorithms: list[str], allowed_models: list[str]
):
    hydra_choices = HydraConfig.get().runtime.choices
    model_name = hydra_choices.model.split("/")[0]
    critic_model_name = hydra_choices["model@critic_model"].split("/")[0]

    experiment_config = ExperimentConfig.get_from_yaml()
    update_config_(experiment_config, cfg.experiment)

    if not Path(experiment_config.save_folder).exists():
        os.makedirs(experiment_config.save_folder)

    algorithm_configs = []
    tasks = []
    for key, value in hydra_choices.items():
        if not key.startswith("algorithm@") and not key.startswith("task@"):
            continue
        iter_type, iter_id = key.split("@")
        iter_name = value.split("/")[0]
        iter_cfg = cfg[iter_id]

        if iter_type == "algorithm":
            algorithm_configs.append(validate_algorithm(iter_name, allowed_algorithms, iter_cfg))
        elif iter_type == "task":
            tasks.append(validate_task(iter_name, allowed_tasks, iter_cfg))

    model_config = validate_model(model_name, allowed_models, cfg.model)
    critic_model_config = validate_model(critic_model_name, allowed_models, cfg.critic_model)

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        model_config=model_config,
        critic_model_config=critic_model_config,
        experiment_config=experiment_config,
        seeds=cfg.seeds,
    )
    return benchmark


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
