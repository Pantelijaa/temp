import cv2
import numpy as np

MIN_AREA = 100

def largest_blob(mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    if areas[biggest - 1] < MIN_AREA:
        return np.zeros_like(mask)

    return (labels == biggest).astype("uint8") * 255

def convex_hull(blob):
    points = cv2.findNonZero(blob)
    if points is None:
        return None
    return cv2.convexHull(points)

def draw_hull(L, hull):
    view = cv2.cvtColor(L.astype("uint8"), cv2.COLOR_GRAY2BGR)
    if hull is not None:
        cv2.polylines(view, [hull], True, (0, 255, 0), 1)
    return view
