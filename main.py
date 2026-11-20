import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from shadow_remover import prepare_reference, remove_shadows, draw_without_shadows

VIDEO = "test_videos/busStation.avi"
SHAKE = False
MAX_SHAKE = 20 # Works only if SHAKE = True
MIN_PEAK = 0.15
MIN_GAIN = 1.3
THRESHOLD = 30
MIN_AREA = 100
MIN_STATIC_FRAME = 25
EVIDENCE_DECAY = 3

#################################################################

def select_roi(capture):
    ok, first = capture.read()
    roi = cv2.selectROI("ROI", first)
    cv2.destroyWindow("ROI")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return roi

def draw_roi(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#################################################################

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

############################################################################################

def shift_frame(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def shake_offsets(seed=42):
    rng = np.random.default_rng(seed)
    sx, sy = 0, 0
    while True:
        yield sx, sy
        sx = int(np.clip(sx + rng.integers(-2, 3), -MAX_SHAKE, MAX_SHAKE))
        sy = int(np.clip(sy + rng.integers(-2, 3), -MAX_SHAKE, MAX_SHAKE))

def no_shake():
    while True:
        yield 0, 0

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

##########################################################################################

def difference_image(L, reference): #FdImage
    return abs(L - reference)

def print_histogram(diff, t):
    counts, _ = np.histogram(diff, bins=np.arange(0, 257))
    values = np.arange(256)
    above = values > t
    colors = np.where(above, "#eb6834", "#2a78d6")

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar(values, counts, width=1.0, color=colors, edgecolor="#fcfcfb", linewidth=0.5)

    ax.axvline(t + 0.5, color="#52514e", linestyle="--", linewidth=1)
    ax.text(t + 2, 0.97, f"T = {t}", color="#52514e", va="top",
            transform=ax.get_xaxis_transform())

    ax.set_yscale("log")
    ax.set_xlim(-1, max(float(diff.max()), t) + 10)
    ax.set_xlabel("difference from reference (brightness levels)", color="#52514e")
    ax.set_ylabel("number of pixels (log scale)", color="#52514e")
    ax.set_title("Histogram of the difference image", color="#0b0b0b", loc="left")

    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    ax.legend(handles=[
        Patch(color="#2a78d6", label=f"background: {counts[~above].sum()} pixels"),
        Patch(color="#eb6834", label=f"foreground: {counts[above].sum()} pixels"),
    ], frameon=False, labelcolor="#52514e", loc="upper right", bbox_to_anchor=(1.0, 0.9))

    fig.tight_layout()
    plt.show()

def threshold_mask(diff, t): #FgMask
    return (diff > t).astype("uint8") * 255

#########################################################################################

def create_mog():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

def mog_mask(mog, L): #BS-FgMask
    return mog.apply(L.astype("uint8"))

#########################################################################################

def static_mask(diff_mask, moving_mask): # FgMask-BS-FgMask
    return cv2.bitwise_and(diff_mask, cv2.bitwise_not(moving_mask))

#########################################################################################

def cleanup(mask):
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, small)
    return cv2.dilate(eroded, big)

def largest_blob(mask, min_area):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    if areas[biggest - 1] < min_area:
        return np.zeros_like(mask)

    return (labels == biggest).astype("uint8") * 255

#################################################################################

def update_evidence(evidence, static):
    evidence += np.where(static > 0, 1, -EVIDENCE_DECAY).astype("int16")
    np.clip(evidence, 0, 2 * MIN_STATIC_FRAME, out=evidence)
    return evidence

def persistent_mask(evidence, static):
    return ((evidence >= MIN_STATIC_FRAME) & (static > 0)).astype("uint8") * 255

##################################################################

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

################################################################

def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]

def read_frames(capture):
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame

def to_luminance(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return lab_img[:, :, 0].astype("float32")

def run():
    capture = cv2.VideoCapture(VIDEO)
    roi = select_roi(capture)
    start = roi
    reference = None
    window = None
    mog = create_mog()

    offsets = shake_offsets() if SHAKE else no_shake()

    for frame, (sx, sy) in zip(read_frames(capture), offsets):
        frame = shift_frame(frame, sx, sy)

        if reference is None:
            ref_patch = crop(frame, roi).copy()
            reference = to_luminance(ref_patch)
            evidence = np.zeros(reference.shape, "int16")
            window = hanning_window(reference.shape)
            shadow_ref = prepare_reference(ref_patch, reference)

        roi, peak = track_roi(reference, frame, roi, window)
        patch = crop(frame, roi)
        L = to_luminance(patch)
        diff = difference_image(L, reference)
        diff_mask = threshold_mask(diff, THRESHOLD)
        diff_mask, shadow = remove_shadows(diff_mask, patch, L, shadow_ref)
        no_shadows = draw_without_shadows(patch, ref_patch, shadow)
        moving_mask = mog_mask(mog, L)
        static = static_mask(diff_mask, moving_mask)
        evidence = update_evidence(evidence, static)
        static = persistent_mask(evidence, static)
        cleaned = cleanup(static)
        blob = largest_blob(cleaned, MIN_AREA)
        hull = convex_hull(blob)

        error = (roi[0] - (start[0] + sx), roi[1] - (start[1] + sy))
        if error != (0, 0):
            print("off by", error, "peak", round(peak, 2))

        draw_roi(frame, roi)
        cv2.imshow("frame", frame)
        cv2.imshow("L", draw_hull(L, hull))
        cv2.imshow("no shadows", no_shadows)
        cv2.imshow("diff mask", diff_mask)
        cv2.imshow("moving mask", moving_mask)
        cv2.imshow("shadow", shadow)
        cv2.imshow("static_mask", static)
        cv2.imshow("cleaned", cleaned)
        cv2.imshow("object", blob)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('h'):
            print_histogram(diff, THRESHOLD)


    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
