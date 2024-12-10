# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Set

import numpy as np


class StateAction:
    def __init__(self, position: np.ndarray, waypoint: Any):
        self.position = position
        self.waypoint = waypoint

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.waypoint}"
        return hash(string_repr)


class AcyclicEnforcer:
    history: Set[StateAction] = set()

    def reset(self):
        self.history = set()
    
    def check_cyclic(self, position: np.ndarray, waypoint: Any, threshold: float) -> bool:
        for item in self.history:
            if (np.linalg.norm(item.waypoint - waypoint) <= threshold and 
                np.linalg.norm(item.position - position) <= threshold):
                return True
            
        return False

    def add_state_action(self, position: np.ndarray, waypoint: Any) -> None:
        state_action = StateAction(position, waypoint)
        self.history.add(state_action)