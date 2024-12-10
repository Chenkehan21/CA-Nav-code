import os
import cv2
import numpy as np
import torch.nn as nn
from typing import List


class HistoryMap(nn.Module):
    def __init__(self, config, full_map_shape) -> None:
        super().__init__()
        self.config = config
        self.shape = full_map_shape
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
        self.history_map = np.ones(self.shape)
    
    def reset(self):
        self.trajectory = np.zeros(self.shape)
        self.history_map = np.ones(self.shape)
        
    def _draw_polyline(self, trajectory_points: List, thickness: int=20):
        image = np.zeros(self.shape)
        points_array = np.array(trajectory_points, dtype=np.int32)
        color = (255, 255, 255)
        is_closed = False
        
        # thicness = 20 => 1.0(meter) takes 20 grids in map
        trajectory = cv2.polylines(image, [points_array], is_closed, color, thickness)
        
        return trajectory.astype(bool)
    
    def forward(self, trajectory_points: List, step: int, current_episode_id: int, alpha: float=0.95):
        if len(trajectory_points) == 2 and trajectory_points[0] == trajectory_points[1]:
            return self.history_map
        trajectory = self._draw_polyline(trajectory_points)
        self.history_map[trajectory == True] *= alpha
        
        if self.visualize:
            self._visualize(self.history_map, step, current_episode_id)
        
        return self.history_map

    def _visualize(self, history_map, step, current_episode_id):
        history_map_vis = history_map.copy()
        history_map_vis[history_map_vis == 0] = 1
        history_map_vis = np.flipud((history_map * 255).astype(np.uint8))
        cv2.imshow("history map", history_map_vis)
        cv2.waitKey(1)
        
        if self.print_images:
            save_dir = os.path.join(self.config.RESULTS_DIR, "history_map/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, history_map_vis)