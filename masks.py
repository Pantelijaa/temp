import cv2
import numpy as np

MIN_STATIC_FRAMES = 25
EVIDENCE_DECAY = 3

def difference_image(L, reference): #FdImage
    return abs(L - reference)

def create_mog():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

def mog_mask(mog, L): #BS-FgMask
    return mog.apply(L.astype("uint8"))

def static_mask(diff_mask, moving_mask): # FgMask-BS-FgMask
    return cv2.bitwise_and(diff_mask, cv2.bitwise_not(moving_mask))

def update_evidence(evidence, static):
    evidence += np.where(static > 0, 2, -EVIDENCE_DECAY).astype("int16")
    np.clip(evidence, 0, 1 * MIN_STATIC_FRAMES, out=evidence)
    return evidence

def persistent_mask(evidence, static):
    return ((evidence >= MIN_STATIC_FRAMES) & (static > 0)).astype("uint8") * 255

def cleanup(mask):
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, small)
    return cv2.dilate(eroded, big)
