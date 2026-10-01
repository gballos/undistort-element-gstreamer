"""Parameter file for nvundistort (the element and the nvivafilter library).

The kernels evaluate OpenCV's initUndistortRectifyMap model per pixel on the
GPU, so all they need is the input camera matrix K, the distortion
coefficients, and the output camera matrix new_K. new_K comes from
cv2.getOptimalNewCameraMatrix(alpha) and is computed here, once.

    python3 tools/undistort_params.py calibration/imx708_intrinsics.npz --alpha 0.0

The .npz holds an OpenCV calibration: camera_matrix (3x3), dist_coeffs (up to
14, OpenCV order), image_size (width, height) and optionally model ("fisheye"
is refused). The frames to undistort must be exactly the frames the
calibration was shot on -- same resolution AND orientation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

_KEYS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")


def load_intrinsics(npz_path, alpha=0.0):
    """(K, dist8, new_K, (w, h)) for the kernels' model, validated.

    Raises for what the kernel cannot represent rather than approximating it:
    the fisheye model, camera-matrix skew, and the thin-prism/tilt terms
    (coefficients 9-14).
    """
    data = np.load(npz_path)
    model = str(data["model"]) if "model" in data.files else "standard"
    if model == "fisheye":
        raise ValueError(f"{npz_path}: fisheye model is not supported by nvundistort")
    K = data["camera_matrix"].astype(np.float64)
    if K[0, 1] != 0.0:
        raise ValueError(f"{npz_path}: camera matrix has skew, not supported by nvundistort")
    d = data["dist_coeffs"].astype(np.float64).ravel()
    if np.any(d[8:] != 0.0):
        raise ValueError(f"{npz_path}: thin-prism/tilt coefficients are not supported by nvundistort")
    d8 = np.zeros(8)
    d8[:min(8, d.size)] = d[:8]
    w, h = (int(v) for v in data["image_size"])
    new_K, _roi = cv2.getOptimalNewCameraMatrix(K, d8, (w, h), alpha, (w, h))
    return K, d8, new_K, (w, h)


def write_params(npz_path, alpha=0.0, out=None):
    """Write the parameter file (default: next to the .npz as
    `<stem>_nvundistort.txt`) and return its path."""
    K, d8, new_K, (w, h) = load_intrinsics(npz_path, alpha)
    npz_path = Path(npz_path)
    out = Path(out) if out else npz_path.with_name(npz_path.stem + "_nvundistort.txt")
    rows = [("width", w), ("height", h),
            ("fx", K[0, 0]), ("fy", K[1, 1]), ("cx", K[0, 2]), ("cy", K[1, 2]),
            ("new_fx", new_K[0, 0]), ("new_fy", new_K[1, 1]),
            ("new_cx", new_K[0, 2]), ("new_cy", new_K[1, 2]),
            *zip(_KEYS, d8)]
    out.write_text(f"# nvundistort params from {npz_path.name}, alpha={alpha}.\n"
                   + "".join(f"{k} {float(v)!r}\n" for k, v in rows))
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", help="calibration .npz")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="0 = crop to valid pixels, 1 = keep full FOV (default: %(default)s)")
    parser.add_argument("-o", "--out", help="output path (default: next to the .npz)")
    args = parser.parse_args()
    print(write_params(args.npz, args.alpha, args.out))


if __name__ == "__main__":
    main()
