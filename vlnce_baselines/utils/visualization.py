import cv2
import numpy as np
from collections import Sequence
from vlnce_baselines.utils.constant import legend_color_palette


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image():
    vis_image = np.ones((755, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    # text = "Observations (Goal: {})".format(goal_name)
    text = "Goal: "
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    # Obeservation: (640, 480); semantic map: (480, 480)
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    
    vis_image = add_class(vis_image, 0, "out of map", legend_color_palette)
    vis_image = add_class(vis_image, 1, "obstacle", legend_color_palette)
    vis_image = add_class(vis_image, 2, "free space", legend_color_palette)
    vis_image = add_class(vis_image, 3, "agent trajecy", legend_color_palette)
    vis_image = add_class(vis_image, 4, "waypoint", legend_color_palette)

    return vis_image


def add_class(vis_image, id, name, color_palette):
    text = f"{id}:{name}"
    font_color = (0, 0, 0)
    class_color = list(color_palette[id])
    class_color.reverse() # BGR -> RGB
    fontScale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = 20, 50
    start_x, start_y = 25, 540 # x is horizontal and point to right, y is vertical and point to down
    gap_x, gap_y = 280, 30
    
    x1 = start_x + (id % 4) * gap_x
    y1 = start_y + (id // 4) * gap_y
    x2 = x1 + w
    y2 = y1 + h
    text_x = x2 + 10
    text_y = (y1 + y2) // 2 + 5
    
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), class_color, thickness=cv2.FILLED)
    vis_image = cv2.putText(vis_image, text, (text_x, text_y),
                            font, fontScale, font_color, thickness, cv2.LINE_AA)
    
    return vis_image


def add_text(image: np.ndarray, text: str, position: Sequence):
    textsize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
    x, y = position
    x += (480 - textsize[0]) // 2
    image = cv2.putText(image, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=0, thickness=1, lineType=cv2.LINE_AA)
    
    return image