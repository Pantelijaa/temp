import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

def plot_histogram(diff, t):
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
