import cv2

def read_frames(capture):
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame

def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]

def to_luminance(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return lab_img[:, :, 0].astype("float32")

def select_roi(capture):
    ok, first = capture.read()
    roi = cv2.selectROI("ROI", first)
    cv2.destroyWindow("ROI")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return roi

def draw_roi(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
