import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist
from vlnce_baselines.utils.acyclic_enforcer import AcyclicEnforcer
from vlnce_baselines.utils.map_utils import get_nearest_nonzero_waypoint


class WaypointSelector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.resolution = config.MAP.MAP_RESOLUTION
        self.distance_threshold = 0.5 * 100 / self.resolution
        
        self.reset()
        
    def reset(self) -> None:
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        self._stick_current_waypoint = False
        self._acyclic_enforcer = AcyclicEnforcer()
        
    def closest_point(self, points: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points - target_point, axis=1)
        
        return points[np.argmin(distances)]
    
    def _get_value(self, position: np.ndarray, value_map: np.ndarray) -> float:
        x, y = position
        value = value_map[x - 5 : x + 6, y - 5: y + 6]
        value = np.mean(value[value != 0])
        
        return value
        
    def forward(self, sorted_waypoints: np.ndarray, frontiers: np.ndarray, position: np.ndarray, 
                collision_map: np.ndarray, value_map: np.ndarray):
        best_waypoint, best_value = None, None
        invalid_waypoint = False
        print("last waypoint: ", self._last_waypoint, self._last_value)
        if not np.array_equal(self._last_waypoint, np.zeros(2)):
            if np.sum(collision_map) > 0:
                """ 
                check if the last_waypoint is too close to the current collision area
                """
                nonzero_indices = np.argwhere(collision_map != 0)
                distances = cdist([self._last_waypoint], nonzero_indices)
                if np.min(distances) <= 5:
                    invalid_waypoint = True
                    print("################################################ close to collision")
                
            if np.sum(frontiers) > 0:
                nonzero_indices = np.argwhere(frontiers != 0)
                distances = cdist([self._last_waypoint], nonzero_indices)
                if np.min(distances) >= 5 * 100 / self.resolution:
                    invalid_waypoint = True
                    print("################################################ too far from frontiers")
                    
            if np.linalg.norm(self._last_waypoint - position) <= self.distance_threshold:
                """ 
                already achieved last waypoint, need to change another waypoint
                """
                invalid_waypoint = True
                print("################################################ achieve")
        
            if invalid_waypoint:
                idx = 0
                new_waypoint = sorted_waypoints[idx]
                while (np.linalg.norm(new_waypoint - position) < self.distance_threshold and 
                       idx < len(sorted_waypoints)):
                    idx += 1
                    new_waypoint = sorted_waypoints[idx]
                self._last_waypoint = new_waypoint
                print("################################################ get new last waypoint")
                
            """ 
            do not change waypoint if last waypoint's value is not getting too worse
            """
            curr_value = self._get_value(self._last_waypoint, value_map)
            # curr_value = value_map[self._last_waypoint[0], self._last_waypoint[1]]
            # if curr_value - self._last_value > -0.01:
            #     best_waypoint = self._last_waypoint
            #     best_value = curr_value
                
            if (np.linalg.norm(self._last_waypoint - position) > self.distance_threshold and 
                (curr_value - self._last_value > -0.03)):
                best_waypoint = self._last_waypoint
            else:
                print("!!!!!!!! already achieve last waypoint")
        
        if best_waypoint is None:
            for waypoint in sorted_waypoints:
                cyclic = self._acyclic_enforcer.check_cyclic(position, waypoint, 
                                                             threshold=0.5*100/self.resolution)
                if cyclic or np.linalg.norm(waypoint - position) <= self.distance_threshold:
                    continue
                
                best_waypoint= waypoint
                break
        
        if best_waypoint is None:
            print("All waypoints are cyclic! Choosing the closest one.")
            best_waypoint = self.closest_point(sorted_waypoints, position)
            
        if value_map[best_waypoint[0], best_waypoint[1]] == 0:
            best_waypoint = get_nearest_nonzero_waypoint(value_map, best_waypoint)
        
        best_value = self._get_value(best_waypoint, value_map)
        self._acyclic_enforcer.add_state_action(position, best_waypoint)
        self._last_value = best_value
        self._last_waypoint = best_waypoint
        
        return best_waypoint, best_value, sorted_waypoints