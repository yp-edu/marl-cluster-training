from benchmarl.environments.pettingzoo.common import PettingZooClass

from benchmarl.environments import Task


class PettingZooKazClass(PettingZooClass):
    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_state(self) -> bool:
        return True


class PettingZooKazTask(Task):
    """Enum for PettingZoo tasks."""

    KAZ = None

    @staticmethod
    def associated_class():
        return PettingZooKazClass
