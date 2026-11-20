# Reference: https://arxiv.org/pdf/1304.1233 Section 3.5
import cv2
import numpy as np

V_LOW = 0.4
V_HIGH = 0.9
MAX_SAT_RISE = 15
MIN_GRADIENT = 12
MAX_ANGLE = np.pi / 10
MIN_AGREEMENT = 0.1
MIN_EDGE_PIXELS = 5

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

def gradient(L):
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2), np.arctan2(gy, gx)

def shadow_candidates(hsv, ref_hsv, diff_mask):
    ratio = hsv[:, :, 2] / (ref_hsv[:, :, 2] + 1.0)
    darker = (ratio > V_LOW) & (ratio < V_HIGH)
    not_more_colorful = (hsv[:, :, 1] - ref_hsv[:, :, 1]) <= MAX_SAT_RISE
    return (darker & not_more_colorful & (diff_mask > 0)).astype("uint8")

def shadow_mask(candidates, L, ref_mag, ref_ang):
    count, labels = cv2.connectedComponents(candidates)
    mag, ang = gradient(L)

    edge = (mag > MIN_GRADIENT) & (ref_mag > MIN_GRADIENT) & (candidates > 0)
    turn = abs(ang - ref_ang)
    # proveriti da li treba oba da bi se clampovao
    turn = np.minimum(turn, 2*np.pi - turn)
    agree = turn < MAX_ANGLE

    edge_labels = labels[edge]
    total = np.bincount(edge_labels, minlength=count)
    matching = np.bincount(edge_labels, weights=agree[edge] ,minlength=count)
    agreement = np.where(total >= MIN_EDGE_PIXELS, matching / np.maximum(total, 1), 1.0)

    is_shadow = agreement > MIN_AGREEMENT
    is_shadow[0] = False
    return is_shadow[labels].astype("uint8") * 255

def prepare_reference(ref_patch, reference):
    ref_hsv = to_hsv(ref_patch)
    ref_mag, ref_ang = gradient(reference)
    return ref_hsv, ref_mag, ref_ang

def remove_shadows(diff_mask, patch, L, shadow_ref):
    ref_hsv, ref_mag, ref_ang = shadow_ref
    candidates = shadow_candidates(to_hsv(patch), ref_hsv, diff_mask)
    shadow = shadow_mask(candidates, L, ref_mag, ref_ang)
    return cv2.bitwise_and(diff_mask, cv2.bitwise_not(shadow)), shadow

