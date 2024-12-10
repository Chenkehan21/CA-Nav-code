"""
The skeleton map is developed for agent to choose a proper path
according to the instruction. The direction part of instruction
such as turn right/left is crucial for navigation. The skeletonized
map can be useful for choosing a right branch
"""

import os
import cv2
import numpy as np
import torch.nn as nn
from numpy import linalg as LA
from collections import defaultdict
from typing import Tuple, Dict, List
from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks

from vlnce_baselines.utils.map_utils import angle_between_vectors

class SkeletonMap(nn.Module):
    def __init__(self, config) -> None:
        super(SkeletonMap, self).__init__()
        self.config = config
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
    
    def detect_branches(self, 
                        skeleton: np.ndarray, 
                        min_distance: int=3,
                        break_radius: int=2) -> Tuple[np.ndarray, np.ndarray]:
        
        # detect intersections first
        corners = corner_harris(skeleton)
        corner_coords = corner_peaks(corners, min_distance)
        
        # break roads at intersections
        skeleton_branches = skeleton.copy()
        radius = break_radius
        for corner in corner_coords:
            start_x = max(0, corner[0] - radius)
            start_y = max(0, corner[1] - radius)
            end_x = min(skeleton.shape[0], corner[0] + radius + 1)
            end_y = min(skeleton.shape[1], corner[1] + radius + 1)
            skeleton_branches[start_x:end_x, start_y:end_y] = 0
        
        return skeleton_branches, corner_coords
        
    def calculate_lines(self, 
                        skeleton_branches: np.ndarray, 
                        corner_coords: np.ndarray,
                        detect_line_radius: int=4,
                        length_threshold: int=2) -> Dict[int, List]:
        
        r"""This method tries to turn all skelton branches
        to straight lines, so that we can use vector to 
        represent each branch which is helpful to calculate
        the branches' degrees.
        
        In order to fiuger out the connectivity between these
        intersection points, we use a circel detection field
        at each point to detect which line is firstly detected
        
        Args: 
            skeleton_branches: a 2D image the same as skeleton image but is cut
                               at each intersection
            corner_coordds: a 2D array including all intersections' coordinates
            detect_line_radius: 
        """
        _, labels = cv2.connectedComponents(skeleton_branches.astype(np.uint8))
        lines_points = defaultdict(list)
        n = -1
        for label in range(1, labels.max() + 1):
            line = labels == label
            
            # filter branches that are too short
            if line.sum() < length_threshold:
                continue
            else:
                n += 1
                for node in corner_coords[:]:
                    radius = detect_line_radius
                    start_x = max(0, node[0] - radius)
                    start_y = max(0, node[1] - radius)
                    end_x = min(skeleton_branches.shape[0], node[0] + radius + 1)
                    end_y = min(skeleton_branches.shape[1], node[1] + radius + 1) 
                    
                    mask = np.zeros_like(line)
                    mask[start_x:end_x, start_y:end_y] = True
                    if (mask * line).sum() > 0:
                        lines_points[n].append(node)
                        lines_points[n].append(line.sum())
                    
        return lines_points

    def line_analyze(self, line_points: Dict[int, List]) -> np.ndarray:
        n = len(line_points)
        angle_matrix = np.zeros((n, n))
        
        for key, line in line_points.items():
            point_coord, _ = line
            start_point, end_point = point_coord
            direction_vector = end_point - start_point
            for other_key, other_line in line_points.items():
                if key == other_key:
                    continue
                other_start_point, other_end_point = other_line
                if np.array_equal(end_point, other_start_point) or np.array_equal(end_point, other_end_point):
                    adjacent_vector = other_end_point - other_start_point
                    angle = angle_between_vectors(direction_vector, adjacent_vector)
                    angle_matrix[key, other_key] = angle
                elif np.array_equal(start_point, other_start_point) or np.array_equal(start_point, other_end_point):
                    adjacent_vector = other_end_point - other_start_point
                    angle = angle_between_vectors(direction_vector, adjacent_vector)
                    angle_matrix[key, other_key] = angle
        
        return angle_matrix
    
    def forward(self, full_map: np.ndarray, current_episode_id: int, step: int) -> np.ndarray:
        skeleton = skeletonize(full_map)
        skeleton_branches, corner_coords = self.detect_branches(skeleton)
        lines = self.calculate_lines(skeleton_branches, corner_coords)
        angle_matrix = self.line_analyze(lines)
        
        if self.visualize:
            self._visualize(skeleton, current_episode_id, step)
        
        return skeleton, angle_matrix
        
    
    def _visualize(self, skeleton: np.ndarray, current_episode_id: int, step: int):
        skeleton_vis = np.flipud(skeleton.astype(np.uint8) * 255)
        cv2.imshow("skeleton", skeleton_vis)
        if self.print_images:
            save_dir = os.path.join(self.config.RESULTS_DIR, "skeletons/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, skeleton_vis)
    
    
"""
merge lines seems to create many problems

    def create_new_line(self, line1: np.ndarray, line2: np.ndarray):
        start1, end1 = line1
        start2, end2 = line2
        if np.array_equal(start1, start2):
            start, end = end1, end2
        elif np.array_equal(start1, end2):
            start, end = start2, end1
        elif np.array_equal(end1, end2):
            start, end = start1, start2
        elif np.array_equal(end1, start2):
            start, end = start1, end2
            
        return start, end
            
    def merge_lines(self, line_points: Dict[List], 
                    angle_matirx: np.ndarray, 
                    merge_threshold: float=15.0):
        tri_angle_matrix = np.triu(angle_matirx)
        index = np.where((tri_angle_matrix < merge_threshold) & (tri_angle_matrix > 0.0))
        index = np.column_stack(index)
        for idx in index:
            idx[0]
            self.create_new_line()
"""