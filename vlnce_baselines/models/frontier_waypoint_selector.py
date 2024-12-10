import numpy as np
import torch.nn as nn
from typing import List
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.acyclic_enforcer import AcyclicEnforcer


class WaypointSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        
    def reset(self) -> None:
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        
    def forward(self, sorted_waypoints: np.ndarray, sorted_values: List, position: np.ndarray):
        best_waypoint_idx = None
        if not np.array_equal(self._last_waypoint, np.zeros(2)):
            curr_index = None
            
            for idx, waypoint in enumerate(sorted_waypoints):
                if np.array_equal(waypoint, self._last_waypoint):
                    curr_index = idx
                    break
            
            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_waypoints, self._last_waypoint, threshold=0.5)
                if closest_index != -1:
                    curr_index = closest_index
            else:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    best_waypoint_idx = curr_index
        
        if best_waypoint_idx is None:
            for idx, waypoint in enumerate(sorted_waypoints):
                cyclic = self._acyclic_enforcer.check_cyclic(position, waypoint, threshold=0.5*20)
                if cyclic:
                    continue
                best_waypoint_idx = idx
                break
        
        if best_waypoint_idx is None:
            print("All waypoints are cyclic! Choosing the closest one.")
            best_waypoint_idx = max(range(len(sorted_waypoints)), 
                                    key=lambda i: np.linalg.norm(sorted_waypoints[i] - position))
        
        best_waypoint = sorted_waypoints[best_waypoint_idx]
        best_value = sorted_values[best_waypoint_idx]
        self._acyclic_enforcer.add_state_action(position, best_waypoint)
        self._last_value = best_value
        self._last_waypoint = best_waypoint
        
        return best_waypoint, best_value, sorted_waypoints