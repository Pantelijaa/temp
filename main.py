import cv2
import numpy as np

from video import select_roi, draw_roi, crop, read_frames, to_luminance
from tracking import hanning_window, track_roi, anchor_box, follow_anchor
from tracking import shift_frame, shake_offsets, no_shake
from threshold import unimodal_threshold, hysteresis_mask, plot_histogram
from shadow_remover import prepare_reference, remove_shadows, draw_without_shadows
from masks import difference_image, create_mog, mog_mask, static_mask
from masks import update_evidence, persistent_mask, cleanup
from objects import largest_blob, convex_hull, draw_hull

#VIDEO = "test_videos/LeftBag.mpg"
#VIDEO = "test_videos/LeftBox.mpg"
VIDEO = "test_videos/busStation.avi"
SHAKE = True
MAX_SHAKE = 8 # Largest simulated shake, also the margin around the anchor box

def run():
    capture = cv2.VideoCapture(VIDEO)
    roi = select_roi(capture)
    start = roi
    reference = None
    window = None
    mog = create_mog()

    offsets = shake_offsets(MAX_SHAKE) if SHAKE else no_shake()

    for frame, (sx, sy) in zip(read_frames(capture), offsets):
        frame = shift_frame(frame, sx, sy)

        if reference is None:
            ref_patch = crop(frame, roi).copy()
            reference = to_luminance(ref_patch)
            evidence = np.zeros(reference.shape, "int16")
            anchor_start = anchor_box(frame.shape, MAX_SHAKE)
            anchor = anchor_start
            anchor_reference = to_luminance(crop(frame, anchor))
            window = hanning_window(anchor_reference.shape)
            shadow_ref = prepare_reference(ref_patch, reference)

        anchor, peak = track_roi(anchor_reference, frame, anchor, window)
        roi = follow_anchor(start, anchor_start, anchor, frame.shape)
        patch = crop(frame, roi)
        L = to_luminance(patch)
        diff = difference_image(L, reference)
        low = unimodal_threshold(diff)
        diff_mask = hysteresis_mask(diff, low)
        diff_mask, shadow = remove_shadows(diff_mask, patch, L, shadow_ref)
        no_shadows = draw_without_shadows(patch, ref_patch, shadow)
        moving_mask = mog_mask(mog, L)
        static = static_mask(diff_mask, moving_mask)
        evidence = update_evidence(evidence, static)
        static = persistent_mask(evidence, static)
        cleaned = cleanup(static)
        blob = largest_blob(cleaned)
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
            plot_histogram(diff, low)


    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
