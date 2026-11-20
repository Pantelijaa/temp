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

def first_frame(capture):
    ok, frame = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame

def select_roi(frame):
    roi = cv2.selectROI("ROI", frame)
    cv2.destroyWindow("ROI")
    return roi

def draw_roi(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
