import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field

from .elements import Config, Grasp, Placement

@dataclass
class Target:
    obj_name: str
    config: Optional[Config] = field(default_factory=lambda : None)
    grasp: Optional[Grasp] = field(default_factory=lambda : None)
    placement: Optional[Placement] = field(default_factory=lambda : None)
    attr: Tuple[str] = ("config", "grasp", "placement")

    @property
    def target_type(self):
        if self.config is not None:
            return "Config"
        elif self.grasp is not None:
            return "Grasp"
        elif self.placement is not None:
            return "Placement"

@dataclass
class Action:
    name: str
    obj_name: Optional[str] = field(default_factory=lambda :None)
    placeable_name: Optional[str] = field(default_factory=lambda :None)
    #config: Optional[np.ndarray] = field(default_factory=lambda :None)
    target: Optional[Target] = field(default_factory=lambda :None)
    rev: bool = field(default_factory=lambda :False)
    stage: int = field(default_factory=lambda :-1)




if __name__ == "__main__":
    pass