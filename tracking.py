import cv2
import numpy as np

from video import crop, to_luminance

MIN_PEAK = 0.15
MIN_GAIN = 1.3

def hanning_window(shape):
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def phase_correlation(a, b):
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)

    cross = np.conj(A) * B
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real

    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    h,w = a.shape
    if peak_x > w //2:
        peak_x -= w
    if peak_y > h //2:
        peak_y -= h
    return int(peak_x), int(peak_y), float(corr.max()), float(corr[0, 0])

def clamp_roi(roi, frame_shape):
    x, y, w, h = roi
    H, W = frame_shape[:2]
    x = min(max(x, 0), W - w)
    y = min(max(y, 0), H - h)
    return x, y, w, h

def track_roi(reference, frame, roi, window):
    patch = to_luminance(crop(frame, roi))
    dx, dy, peak, stay = phase_correlation(reference * window, patch * window)
    if peak < MIN_PEAK or peak < MIN_GAIN * stay:
        return roi, peak
    x, y, w, h = roi
    return clamp_roi((x + dx, y + dy, w, h), frame.shape), peak

def anchor_box(frame_shape, margin):
    H, W = frame_shape[:2]
    return margin, margin, W - 2 * margin, H - 2 * margin

def follow_anchor(start, anchor_start, anchor, frame_shape):
    x, y, w, h = start
    dx = anchor[0] - anchor_start[0]
    dy = anchor[1] - anchor_start[1]
    return clamp_roi((x + dx, y + dy, w, h), frame_shape)

# Simulated camera shake, for testing the tracker
def shift_frame(frame, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def shake_offsets(max_shake, seed=42):
    rng = np.random.default_rng(seed)
    sx, sy = 0, 0
    while True:
        yield sx, sy
        sx = int(np.clip(sx + rng.integers(-2, 3), -max_shake, max_shake))
        sy = int(np.clip(sy + rng.integers(-2, 3), -max_shake, max_shake))

def no_shake():
    while True:
        yield 0, 0
