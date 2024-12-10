import cv2
import torch
import numpy as np
from typing import Tuple, List
import torch.nn.functional as F
from collections import Sequence
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_objects, closing, disk, dilation

from vlnce_baselines.utils.constant import *
from vlnce_baselines.utils.pose import get_agent_position, threshold_poses
import time


def get_grid(pose: torch.Tensor, grid_size: Tuple, device: torch.device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180. # t.shape=[bs]
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1) # [bs, 3]
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1) # [bs, 2, 3] rotation matrix

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1) # [bs, 2, 3] translation matrix

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1 # size=11
    mask = np.zeros((size, size)) # (11,11)
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2 \
               and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 > \
                    (step_size - 1) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size  ** 2:
                mask[i, j] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


def create_sector_mask(position: Sequence, heading: float, radius: float,
                       angle: float, map_shape: Sequence):
    """ 
    arg "position" came from full pose, full pose use standard Cartesian coordinate.
    """
    mask = np.zeros(map_shape)
    heading = (360 - heading) % 360
    angle_high = (heading + angle / 2) % 360
    angle_low = (heading - angle / 2) % 360

    y, x = np.meshgrid(np.arange(map_shape[0]) - position[0], np.arange(map_shape[1]) - position[1])
    distance = np.sqrt(x**2 + y**2)
    angle = np.arctan2(x, y) * 180 / np.pi
    angle = (360 - angle) % 360

    valid_distance = distance <= radius
    if angle_high > angle_low:
        valid_angle = (angle_low <= angle) & (angle <= angle_high)
    else:
        valid_angle = (angle_low <= angle) | (angle <= angle_high)
    mask[valid_distance & valid_angle] = 1

    return mask


def get_collision_mask(known_vector: np.ndarray, mask_data: np.ndarray, angle_threshold: float):
    collision_map = np.zeros_like(mask_data)
    center = np.array(mask_data.shape) // 2

    rows, cols = np.indices(mask_data.shape)
    rows_from_center = rows - center[0]
    cols_from_center = cols - center[1]

    nonzero_indices = np.nonzero(mask_data)

    vectors = np.array([rows_from_center[nonzero_indices], cols_from_center[nonzero_indices]])
    vector_lengths = np.linalg.norm(vectors, axis=0)
    known_vector_length = np.linalg.norm(known_vector)
    rotation_matrix = np.array([[0, -1], 
                                [1, 0]])
    known_vector = np.dot(rotation_matrix, known_vector)
    known_vector_expanded = np.tile(known_vector[:, np.newaxis], vectors.shape[1])
    
    cos_angles = np.sum(known_vector_expanded * vectors, axis=0) / (known_vector_length * vector_lengths + 1e-10)
    angles_rad = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    angles_deg = np.degrees(angles_rad)
    print("angles: ", angles_deg)
    collision_map[nonzero_indices[0][angles_deg <= angle_threshold], 
                  nonzero_indices[1][angles_deg <= angle_threshold]] = 1

    return collision_map


def process_navigable_classes(classes: List):
    classes = [item.strip().lower() for item in classes]
    common_items = set(navigable_classes) & set(classes)
    if len(common_items) > 0:
        navigable_index = [classes.index(item) for item in common_items]
    else:
        navigable_index = []
    
    return navigable_index


def get_obstacle(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    """
    The agent radius is 0.18m and resolution is 5, so the agent
    takes at least a 8*8 square area in map whose size is 64.
    Now we will remove some small objects(think of them as noise) first
    and then do morphological closing, so set min_size=64 which is coincidentally
    the default value of min_size is a good choice
    """
    obstacle = map[0, ...]
    obstacle = remove_small_objects(
        obstacle.astype(bool), 
        min_size=64, # you can try different minimum object size
        connectivity=5)
    selem = disk(kernel_size)
    obstacle = closing(obstacle, selem)
    
    return obstacle.astype(bool)


def get_objects(map: np.ndarray, classes: List, kernel_size: int=3) -> Tuple:
    navigable = np.zeros(map.shape[-2:])
    navigable_index = process_navigable_classes(classes)
    objects = np.zeros(map.shape[-2:])
    for i, obj in enumerate(map[map_channels:, ...]):
        obj = remove_small_objects(obj.astype(bool), min_size=64)
        obj = closing(obj, selem=disk(kernel_size))
        if i in navigable_index:
            navigable = np.logical_or(navigable, obj)
        else:
            objects = np.logical_or(obj, objects)
    
    return objects.astype(bool), navigable.astype(bool)


def get_explored_area(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    """ 
    when extract large area like explored area, we prefer do morphological
    closing first then remove small objects.
    the agent takes 8*8 area and one step is 0.25m which takes 5 squares
    so an area of 20*20 size is enough for the agent to take one step in four directions.
    """
    explored_area = map[1, ...]
    selem = disk(kernel_size)
    explored_area = closing(explored_area, selem)
    explored_area = remove_small_objects(explored_area.astype(bool), min_size=400)
    
    return explored_area


def process_floor(map: np.ndarray, classes: List, kernel_size: int=3) -> np.ndarray:
    """
    we didn't use get_objects() and get_explored_area() here because
    we want to extract the floor area more precisely. So we're going 
    to do the morphological closing at the last step.
    """
    t1 = time.time()
    navigable_index = process_navigable_classes(classes)
    # print("classes: ", classes)
    # print("navigable index: ", navigable_index)
    # print("map length: ", len(map))
    navigable = np.zeros(map.shape[-2:])
    explored_area = map[1, ...]
    obstacles = map[0, ...]
    objects = np.zeros(map.shape[-2:])
    for i, obj in enumerate(map[map_channels:, ...]):
        if i in navigable_index:
            navigable += obj
        else:
            objects += obj
    free_mask = 1 - np.logical_or(obstacles, objects)
    free_mask = np.logical_or(free_mask, navigable)
    free_space = explored_area * free_mask
    floor = remove_small_objects(free_space.astype(bool), min_size=400)
    floor = closing(floor, selem=disk(kernel_size))
    # floor = remove_small_objects(free_space.astype(bool), min_size=400)
    t2 = time.time()
    print("process floor cost time: ", t2 - t1)
    return floor


def find_frontiers(map: np.ndarray, classes: List) -> np.ndarray:
    # floor = process_floor(map)
    floor = get_floor_area(map, classes)
    explored_area = get_explored_area(map)
    contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.zeros(map.shape[-2:], dtype=np.uint8)
    image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
    res = np.logical_and(floor, image)
    res = dilation(res, selem=disk(2))
    res = remove_small_objects(res.astype(bool), min_size=64)
    
    return res.astype(np.uint8)


def get_traversible_area(map: np.ndarray, classes: List) -> np.ndarray:
    """ 
    Sometimes there may be holes in enclosed obstacle areas
    this function aims to fill these holes.
    """
    objects, navigable = get_objects(map, classes)
    obstacles = get_obstacle(map)
    untraversible = np.logical_or(objects, obstacles)
    untraversible[navigable == 1] = 0
    untraversible = remove_small_objects(untraversible, min_size=64)
    untraversible = closing(untraversible, selem=disk(3))
    traversible = 1 - untraversible
    # traversible_area = np.sum(traversible)
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(traversible.astype(np.uint8))
    # if nb_components > 2:
    #     areas = [np.sum(output == i) for i in range(1, nb_components)]
    #     left_areas = [traversible_area - item for item in areas]
    #     min_idx = left_areas.index(min(left_areas)) + 1
    #     res = np.ones(map.shape[-2:])
    #     for i in range(nb_components):
    #         if i != min_idx:    
    #             res[output == i] = 0
    #     return res
    # else:
    #     return traversible
    return traversible

def get_floor_area(map: np.ndarray, classes: List) -> np.ndarray:
    """ 
    find traversible area that are connected with floor area
    """
    traversible = get_traversible_area(map, classes)
    floor = process_floor(map, classes)
    res = np.logical_xor(floor, traversible)
    res = remove_small_objects(res, min_size=64)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(res.astype(np.uint8))
    if nb_components > 2:
        areas = [np.sum(output == i) for i in range(1, nb_components)]
        max_id = areas.index(max(areas)) + 1
        for i in range(1, nb_components):
            if i != max_id:
                floor = np.logical_or(floor, output==i)
                
    return floor.astype(bool)


def get_nearest_nonzero_waypoint(arr: np.ndarray, start: Sequence) -> np.ndarray:
    nonzero_indices = np.argwhere(arr != 0)
    if len(nonzero_indices) > 0:
        distances = cdist([start], nonzero_indices)
        nearest_index = np.argmin(distances)
        
        return np.array(nonzero_indices[nearest_index])
    else:
        return np.array([int(start[0]), int(start[1])])


def angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    dot_product = np.dot(vector1, vector2)
    vector1_length = np.linalg.norm(vector1)
    vector2_length = np.linalg.norm(vector2)
    angle = np.arccos(dot_product / (vector1_length * vector2_length))
    
    cross_product = np.cross(vector1, vector2)
    if cross_product == 0 and vector1[0] == vector2[0] * -1:
        return 180.
    signed_angle = np.sign(cross_product) * angle
    angle_degrees = np.degrees(signed_angle)
    
    return angle_degrees


def angle_to_vector(angle: float) -> np.ndarray:
    angle_rad = np.radians(angle)
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    
    return np.array([x, y])


def process_destination(destination: np.ndarray, full_map: np.ndarray, classes: List) -> np.ndarray:
    """ 
    destination could be some small objects, so we dilate them first
    and then remove small objects
    """
    floor = process_floor(full_map, classes)
    traversible = get_traversible_area(full_map, classes)
    destination = dilation(destination, selem=disk(5))
    destination = remove_small_objects(destination.astype(bool), min_size=64).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(destination)
    if len(centroids) > 1:
        centroid = centroids[1] # the first one is background
        waypoint = np.array([int(centroid[1]), int(centroid[0])])
        waypoint = get_nearest_nonzero_waypoint(np.logical_and(floor, traversible), waypoint)
        return waypoint
    else:
        return None

def process_destination2(destination: np.ndarray, floor: np.ndarray, traversible: np.ndarray) -> np.ndarray:
    """ 
    destination could be some small objects, so we dilate them first
    and then remove small objects
    """
    destination = dilation(destination, selem=disk(5))
    destination = remove_small_objects(destination.astype(bool), min_size=64).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(destination)
    if len(centroids) > 1:
        centroid = centroids[1] # the first one is background
        waypoint = np.array([int(centroid[1]), int(centroid[0])])
        waypoint = get_nearest_nonzero_waypoint(traversible, waypoint)
        return waypoint
    else:
        return None


def angle_and_direction(a: np.ndarray, b: np.ndarray, turn_angle: float) -> Tuple:
    unit_a = a / (np.linalg.norm(a) + 1e-5)
    unit_b = b / (np.linalg.norm(b) + 1e-5)
    
    cross_product = np.cross(unit_a, unit_b)
    dot_product = np.dot(unit_a, unit_b)
    
    angle = np.arccos(dot_product)
    angle_degrees = np.degrees(angle)
    
    if cross_product > 0 and angle_degrees >= (turn_angle / 2 + 0.01):
        direction = 3 # right
        # print("turn right", angle_degrees)
    elif cross_product < 0 and angle_degrees >= (turn_angle / 2):
        direction = 2 # left
        # print("turn left", angle_degrees)
    elif cross_product == 0 and angle_degrees == 180:
        direction = 3
    else:
        direction = 1 # forward
        # print("go forward", angle_degrees, cross_product)
    
    return angle_degrees, direction


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        points_array (np.ndarray): An array of 2D points, where each point is a tuple
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1
    
    
def collision_check(last_pose: np.ndarray, current_pose: np.ndarray,
                    resolution: float, map_shape: Sequence,
                    collision_threshold: float=0.2,
                    width: float=0.4, height: float=1.5, buf: float=0.2) -> np.ndarray:
    last_position, last_heading = get_agent_position(last_pose, resolution)
    x0, y0 = last_position
    current_position, _ = get_agent_position(current_pose, resolution)
    position_vector = current_position - last_position
    displacement = np.linalg.norm(position_vector)
    collision_map = np.zeros(map_shape)
    print("displacement: ", displacement)
    
    if displacement < collision_threshold * 100 / resolution:
        print("!!!!!!!!! COLLISION !!!!!!!!!")
        theta = np.deg2rad(last_heading)
        width_range = int(width * 100 / resolution)
        height_range = int(height * 100 / resolution)
        # width_range = int((0.25 - displacement) * 100 / resolution * 2)
        # height_range = int((0.25 - displacement) * 100 / resolution * 2)
        buf = displacement * 100 / resolution + 3
        
        for i in range(height_range):
            for j in range(width_range):
                l1 = j + buf
                l2 = i - width_range // 2
                dy = l1 * np.cos(theta) + l2 * np.sin(theta) # change to ndarray coordinate
                dx = l1 * np.sin(theta) - l2 * np.cos(theta) # change to ndarray coordinate
                x1 = int(x0 - dx)
                y1 = int(y0 + dy)
                x1, y1 = threshold_poses([x1, y1], collision_map.shape)
                collision_map[x1, y1] = 1
                
                dy = l1 * np.cos(theta) - l2 * np.sin(theta) # change to ndarray coordinate
                dx = l1 * np.sin(theta) + l2 * np.cos(theta) # change to ndarray coordinate
                x1 = int(x0 - dx)
                y1 = int(y0 + dy)
                x1, y1 = threshold_poses([x1, y1], collision_map.shape)
                collision_map[x1, y1] = 1
        collision_map = closing(collision_map, selem=disk(1))
        
    return collision_map

# def collision_check_fmm(last_pose: np.ndarray, current_pose: np.ndarray,
#                     resolution: float, map_shape: Sequence,
#                     collision_threshold: float=0.2,
#                     width: float=0.4, height: float=1.5, buf: float=0.2) -> np.ndarray:
#     last_position, last_heading = get_agent_position(last_pose, resolution)
#     current_position, current_heading = get_agent_position(current_pose, resolution)
#     x, y = current_position
#     position_vector = current_position - last_position
#     displacement = np.linalg.norm(position_vector)
#     collision_map = np.zeros(map_shape)
#     print("displacement: ", displacement)
    
#     if displacement < collision_threshold * 100 / resolution:
#         print("!!!!!!!!! COLLISION !!!!!!!!!")
#         dx, dy = x - int(x), y - int(y)
#         mask = get_mask(dx, dy, scale=1, step_size=5)
#         direction_mask = create_sector_mask((5, 5), current_pose[-1], 6, 45, (11,11))
#         collision_mask = np.logical_and(mask, direction_mask)
#         x, y = int(x), int(y)
#         print("!!!!!!!!!!!!!!!! collision x and y: ", x, y)
#         collision_map[x - 5 : x + 6, y - 5: y + 6] = collision_mask
#         collision_map[5,5] = 0
        
#         cv2.imshow("direction mask: ", (np.flipud(direction_mask * 255)).astype(np.uint8))
#         cv2.imshow("mask: ", (np.flipud(mask * 255)).astype(np.uint8))
#         cv2.imshow("collision mask: ", (np.flipud(collision_mask * 255)).astype(np.uint8))
#         cv2.waitKey(1)
    
#     return collision_map

def calculate_displacement(last_pose: np.ndarray, current_pose: np.ndarray, resolution: float):
    last_position, last_heading = get_agent_position(last_pose, resolution)
    current_position, current_heading = get_agent_position(current_pose, resolution)
    x, y = current_position
    position_vector = current_position - last_position
    displacement = np.linalg.norm(position_vector)
    
    return displacement

def collision_check_fmm(last_pose: np.ndarray, current_pose: np.ndarray,
                    resolution: float, map_shape: Sequence,
                    collision_threshold: float=0.2,
                    width: float=0.4, height: float=1.5, buf: float=0.2) -> np.ndarray:
    last_position, last_heading = get_agent_position(last_pose, resolution)
    current_position, current_heading = get_agent_position(current_pose, resolution)
    x, y = current_position
    position_vector = current_position - last_position
    displacement = np.linalg.norm(position_vector)
    collision_map = np.zeros(map_shape)
    collision_mask = None
    print("displacement: ", displacement)
    
    if displacement < collision_threshold * 100 / resolution:
        print("!!!!!!!!! COLLISION !!!!!!!!!")
        dx, dy = x - int(x), y - int(y)
        mask = get_mask(dx, dy, scale=1, step_size=5)
        heading_vector = angle_to_vector(current_heading)
        collision_mask = get_collision_mask(heading_vector, mask, 32)
        x, y = int(x), int(y)
        if x - 5 >= 0 and x + 6 < map_shape[0] and y - 5 >= 0 and y + 6 < map_shape[1]:
            collision_map[x - 5 : x + 6, y - 5 : y + 6] = collision_mask
        # cv2.imshow("mask: ", (np.flipud(mask * 255)).astype(np.uint8))
        # cv2.imshow("collision mask: ", (np.flipud(collision_mask * 255)).astype(np.uint8))
    
    return collision_map
        
    
# def get_traversible_area(map: np.ndarray) -> np.ndarray:
#     objects = get_objects(map)
#     obstacles = get_obstacle(map)
#     traversible = 1 - (objects + obstacles)

#     return traversible


# def get_obstacle(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
#     floor = map[4, ...] # channel is floor area
#     floor_mask = 1 - (floor != 0)
#     obstacle = map[0, ...] * floor_mask
#     obstacle = remove_small_objects(
#         obstacle.astype(bool), 
#         min_size=50, # you can try different minimum object size
#         connectivity=5)
#     selem = disk(kernel_size)
#     obstacle = closing(obstacle, selem)
    
#     return obstacle.astype(bool)


# def get_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
#     floor = map[4, ...]
#     selem = disk(kernel_size)
#     floor = closing(floor, selem)
#     floor = remove_small_objects(floor.astype(bool), min_size=200)
    
#     return floor.astype(bool)


# def process_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
#     floor = get_floor(map, kernel_size)
#     obstacles = get_obstacle(map, kernel_size)
#     objects = get_objects(map, kernel_size)
    
#     obstacles_mask = (obstacles == False)
#     objects_mask = (objects == False)
#     floor = floor * objects_mask * obstacles_mask
#     floor = remove_small_objects(floor.astype(bool), min_size=200)
    
#     return floor