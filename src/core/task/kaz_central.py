from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = "knights_archers_zombies_v10"
    other: bool = MISSING
