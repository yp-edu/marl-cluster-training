from dataclasses import dataclass, MISSING


@dataclass
class KazTaskConfig:
    task: str = MISSING
    spawn_rate: int = MISSING
    num_archers: int = MISSING
    num_knights: int = MISSING
    max_zombies: int = MISSING
    max_arrows: int = MISSING
    killable_knights: bool = MISSING
    killable_archers: bool = MISSING
    pad_observation: bool = MISSING
    line_death: bool = MISSING
    max_cycles: int = MISSING
    vector_state: bool = MISSING
    use_typemasks: bool = MISSING
    sequence_space: bool = MISSING
