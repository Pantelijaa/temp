import numpy as np
import cv2

HIGH_FACTOR = 3

def unimodal_threshold(diff):
    counts, edges = np.histogram(diff, bins=256, range=(0, 256))
    peak = int(np.argmax(counts))
    end = int(np.nonzero(counts)[0][-1]) + 1
    heights = np.append(counts, 0)[peak:end + 1].astype("float64")
    position = np.arange(peak, end + 1)
    distance = heights[0] * (end - position) - (end - peak) * heights
    return float(edges[position[np.argmax(distance)]])

def hysteresis_mask(diff, low):
    high = low * HIGH_FACTOR
    weak = (diff > low).astype("uint8")
    count, labels = cv2.connectedComponents(weak)
    keep = np.bincount(labels[diff > high], minlength=count) > 0
    keep[0] = False
    return keep[labels].astype("uint8") * 255
