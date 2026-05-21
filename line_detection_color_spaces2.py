import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ─────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────

def compute_signal(crop_bgr, method="rgb"):

    crop_rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB
    )

    crop_rgb = crop_rgb.astype(np.float32)

    eps = 1e-6

    if method == "rgb":

        R = crop_rgb[:, :, 0]

        # dark red lines -> high signal
        signal = 255 - R

    elif method == "delta_rgb":

        R = crop_rgb[:, :, 0]
        G = crop_rgb[:, :, 1]
        B = crop_rgb[:, :, 2]

        R0 = np.median(R)
        G0 = np.median(G)
        B0 = np.median(B)

        signal = np.sqrt(
            (R - R0) ** 2 +
            (G - G0) ** 2 +
            (B - B0) ** 2
        )

    elif method == "rgb_bg_corrected":

        R = crop_rgb[:, :, 0]
        G = crop_rgb[:, :, 1]
        B = crop_rgb[:, :, 2]

        I = R + G + B + eps

        r = R / I
        g = G / I
        b = B / I

        # Handles:
        # red lines
        # dark red lines
        # purple-ish lines in dark scenes

        signal = (
            1.5 * r +
            0.7 * b -
            0.5 * g
        )

        signal = signal - np.percentile(signal, 5)

        signal = np.clip(signal, 0, None)

    else:
        raise ValueError(f"Unknown method: {method}")

    signal = signal.astype(np.float32)

    signal /= (np.max(signal) + eps)

    return signal


# ─────────────────────────────────────────────
# PEAK DETECTION
# ─────────────────────────────────────────────
'''
def detect_with_kernel(signal_map, kernel_size, method):

    h, w = signal_map.shape

    # remove dark strip borders
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    projection = np.mean(
        signal_map[:, x1:x2],
        axis=1
    )

    # remove slow illumination drift
    baseline = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 151),
        0
    ).flatten()

    projection = projection - baseline

    projection = projection - np.min(projection)

    # ONLY smooth 1D signal
    smoothed = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, kernel_size),
        0
    ).flatten()

    smoothed /= (
        np.max(smoothed) + 1e-6
    )

    # no padding
    # no border replication
    # shorter signal is fine

    peaks, properties = find_peaks(
        smoothed,
        prominence=0.10,
        distance=80,
        width=8
    )

    return smoothed, peaks, properties
'''

def detect_with_kernel(signal_map, kernel_size, method):

    h, w = signal_map.shape

    # center ROI only
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)

    projection = np.mean(
        signal_map[:, x1:x2],
        axis=1
    )

    # VERY IMPORTANT:
    # stronger baseline smoothing
    baseline = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 201),
        0
    ).flatten()

    projection = projection - baseline

    projection -= np.min(projection)

    # mild smoothing ONLY on 1D profile
    smoothed = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, kernel_size),
        0
    ).flatten()

    # normalize
    smoothed /= (
        np.max(smoothed) + 1e-6
    )

    # ─────────────────────────────────────
    # ADAPTIVE THRESHOLD
    # ─────────────────────────────────────

    mean_signal = np.mean(smoothed)
    std_signal = np.std(smoothed)

    adaptive_height = mean_signal + 1.25 * std_signal

    # prevents tiny bumps
    adaptive_height = max(
        adaptive_height,
        0.35
    )

    peaks, properties = find_peaks(
        smoothed,

        height=adaptive_height,

        prominence=0.18,

        distance=120,

        width=(6, 45)
    )

    return smoothed, peaks, properties

# ─────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────

def detect_lines(
    crop_bgr,
    methods=["rgb"],
    kernel_sizes=[15],
    debug=True
):

    all_results = {}

    best_method = None
    best_kernel = None
    best_count = 0

    best_score = -999

    for method in methods:

        signal_map = compute_signal(
            crop_bgr,
            method
        )

        all_results[method] = {}

        for k in kernel_sizes:

            smoothed, peaks, props = detect_with_kernel(
                signal_map,
                k,
                method
            )

            count = len(peaks)

            peak_score = 0

            if count > 0:
                peak_score = np.sum(
                    props["prominences"]
                )

            all_results[method][k] = {
                "peaks": peaks,
                "count": count,
                "smoothed": smoothed,
                "score": peak_score
            }

            if peak_score > best_score:

                best_score = peak_score

                best_count = count

                best_method = method

                best_kernel = k

    # ─────────────────────────────────────────
    # DEBUG VISUALIZATION
    # ─────────────────────────────────────────

    if debug:

        crop_rgb = cv2.cvtColor(
            crop_bgr,
            cv2.COLOR_BGR2RGB
        )

        for method in methods:

            for k in kernel_sizes:

                data = all_results[method][k]

                vis = crop_rgb.copy()

                for y in data["peaks"]:

                    cv2.line(
                        vis,
                        (0, y),
                        (vis.shape[1], y),
                        (255, 0, 0),
                        2
                    )

                plt.figure(figsize=(10, 4))

                # IMAGE
                plt.subplot(1, 2, 1)

                plt.title(
                    f"{method} | kernel={k} px | lines={data['count']}"
                )

                plt.imshow(vis)

                plt.xlabel("Horizontal position (pixels)")

                plt.ylabel("Vertical position (pixels)")

                # SIGNAL
                plt.subplot(1, 2, 2)

                plt.title("Vertical Intensity Profile")

                x_axis = np.arange(
                    len(data["smoothed"])
                )

                plt.plot(
                    x_axis,
                    data["smoothed"]
                )

                for p in data["peaks"]:

                    plt.axvline(
                        p,
                        color="red",
                        linewidth=1
                    )

                plt.xlabel(
                    "Vertical position (pixels)"
                )

                plt.ylabel(
                    "Normalized signal intensity (a.u.)"
                )

                plt.tight_layout()

                plt.show()

    return (
        best_count,
        best_method,
        best_kernel,
        all_results,
        signal_map
    )