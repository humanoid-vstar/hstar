from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass

@dataclass
class HstarServiceConfig(BaseServiceConfig):
    use_state_reward: bool = False
    max_workers: int = 64

