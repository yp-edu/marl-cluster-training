from hydra.core.config_store import ConfigStore

from benchmarl.environments import PettingZooTask
from benchmarl.environments import task_config_registry

from scripts.kaz.task import PettingZooKazTask
from scripts.kaz.config import KazTaskConfig


def setup_custom_tasks(main):
    cs = ConfigStore.instance()
    cs.store(name="pettingzoo_kaz_config", group="task", node=KazTaskConfig)

    task_config_registry["kaz/default"] = PettingZooKazTask.KAZ
    task_config_registry["multiwalker/shared"] = PettingZooTask.MULTIWALKER

    return main
