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
        self.distance_threshold = 0.25 * 100 / self.resolution
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        self._stick_current_waypoint = False
        self.change_threshold = self.config.EVAL.CHANGE_THRESHOLD
        
    def reset(self) -> None:
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        self._stick_current_waypoint = False
        self._acyclic_enforcer.reset()
        
    def closest_point(self, points: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points - target_point, axis=1)
        
        return points[np.argmin(distances)]
    
    def _get_value(self, position: np.ndarray, value_map: np.ndarray) -> float:
        x, y = position
        value = value_map[x - 5 : x + 6, y - 5: y + 6]
        value = np.mean(value[value != 0])
        
        return value
        
    def forward(self, sorted_waypoints: np.ndarray, position: np.ndarray, collision_map: np.ndarray, 
                value_map: np.ndarray, fmm_dist: np.ndarray, traversible: np.ndarray, replan: bool):
        best_waypoint, best_value = None, None
        invalid_waypoint = False
        # print("last waypoint: ", self._last_waypoint, self._last_value)
        if not np.array_equal(self._last_waypoint, np.zeros(2)):
            if replan:
                invalid_waypoint = True

            if np.sum(collision_map) > 0:
                """ 
                check if the last_waypoint is too close to the current collision area
                """
                nonzero_indices = np.argwhere(collision_map != 0)
                distances = cdist([self._last_waypoint], nonzero_indices)
                if np.min(distances) <= 5:
                    invalid_waypoint = True
                    print("################################################ close to collision")
                    
            if np.linalg.norm(self._last_waypoint - position) < self.distance_threshold:
                invalid_waypoint = True
                print("################################################ achieve")
            
            x, y = int(position[0]), int(position[1])
            if fmm_dist is not None:
                print("fmm dist: ", np.mean(fmm_dist[x-10:x+11, y-10:y+11]), np.max(fmm_dist))
            if fmm_dist is not None and abs(np.mean(fmm_dist[x-10:x+11, y-10:y+11]) - np.max(fmm_dist)) <= 5.0:
                invalid_waypoint = True
                print("################################################ created an enclosed area!")
        
            if invalid_waypoint:
                idx = 0
                new_waypoint = sorted_waypoints[idx]
                distance_flag = np.linalg.norm(new_waypoint - position) < self.distance_threshold
                last_waypoint_flag = np.linalg.norm(new_waypoint - self._last_waypoint) < self.distance_threshold
                flag = distance_flag or last_waypoint_flag
                while ( flag and idx + 1 < len(sorted_waypoints)):
                    idx += 1
                    new_waypoint = sorted_waypoints[idx]
                self._last_waypoint = new_waypoint
                
            """ 
            if last_waypoint's current value doesn't get worse too much 
            then we stick to it.
            """
            curr_value = self._get_value(self._last_waypoint, value_map)
            # if curr_value - self._last_value > -0.01:
            #     best_waypoint = self._last_waypoint
            #     best_value = curr_value
                
            if ((np.linalg.norm(self._last_waypoint - position) > self.distance_threshold) and 
                (curr_value - self._last_value > self.change_threshold)):
                best_waypoint = self._last_waypoint
        
        if best_waypoint is None:
            for waypoint in sorted_waypoints:
                cyclic = self._acyclic_enforcer.check_cyclic(position, waypoint, 
                                                             threshold=0.5*100/self.resolution)
                if cyclic or np.linalg.norm(waypoint - position) < self.distance_threshold:
                    continue
                
                best_waypoint= waypoint
                break
        
        if best_waypoint is None:
            print("All waypoints are cyclic! Choosing the closest one.")
            best_waypoint = self.closest_point(sorted_waypoints, position)
        
        if traversible[best_waypoint[0], best_waypoint[1]] == 0:
            best_waypoint = get_nearest_nonzero_waypoint(traversible, best_waypoint)
            
        best_value = self._get_value(best_waypoint, value_map)
        
        self._acyclic_enforcer.add_state_action(position, best_waypoint)
        self._last_value = best_value
        self._last_waypoint = best_waypoint
        # print("best waypoint: ", best_waypoint)
        
        return best_waypoint, best_value, sorted_waypoints