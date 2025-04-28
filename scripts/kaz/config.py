from dataclasses import dataclass, MISSING


@dataclass
class KazTaskConfig:
    task: str = MISSING
    max_cycles: int = MISSING
