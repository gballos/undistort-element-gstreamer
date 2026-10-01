# Tests

On-Jetson tests for the undistortion: the `nvundistort` element and the
`nvivafilter` library `libnvundistort.so`. Run them from a Jetson (Orin) that
has GStreamer with `nvivafilter`, DeepStream, plus `python3` with
`opencv-python` and `numpy`.

**Build first** (the library needs NVIDIA's `customer_functions.h` in
`native/nvundistort/include/`, see `../native/nvundistort/README.md`):

```bash
make -C ../native/nvundistort           # library, from this tests/ dir
make -C ../native/nvundistort/element   # element
```

The calibration artifacts live at the repo root:

| File | What it is |
|---|---|
| `../imx708_intrinsics.npz` | IMX708 calibration (`camera_matrix`, `dist_coeffs`, `image_size`, `model`) |
| `../imx708_intrinsics_nvundistort.txt` | Its plain `key value` params file, `alpha=0`, used by the library |

The calibration is valid **only** for 4608×2592, no rotation, focus locked. On
the Pi: `rpicam-vid -t 0 --width 4608 --height 2592 --framerate 14 --codec
mjpeg --autofocus-mode manual --lens-position 0 --flush --listen -o
tcp://192.168.10.1:8881`.

## `check_undistort.py` — accuracy

Runs one JPEG through the undistortion and compares against `cv2.remap`.
PASS = mean absolute difference below 1 level on Y, U and V.

```bash
cd tests
python3 check_undistort.py frame.jpg --npz ../imx708_intrinsics.npz --out-dir /tmp            # library
python3 check_undistort.py frame.jpg --npz ../imx708_intrinsics.npz --out-dir /tmp --element  # element
```

If `import cv2` fails with a NumPy 2 error, a user-installed NumPy is shadowing
the system one: run with `PYTHONNOUSERSITE=1`.

`frame.jpg` must be shot at the calibrated resolution and orientation. Outputs
the raw NV12 frames and `/tmp/undistort_preview.jpg` (straight edges should be
straight).

## `undistort_params.py` — regenerate the params file

Turns a calibration `.npz` into the `key value` text file the library reads.
Run it after a new calibration or a change of `alpha`, then commit the result.

```bash
cd tests
python3 undistort_params.py ../imx708_intrinsics.npz --alpha 0.0 -o ../imx708_intrinsics_nvundistort.txt
```

`alpha`: 0 crops to valid pixels, 1 keeps the full field of view with black
borders. It refuses what the kernel cannot represent (fisheye, skew,
coefficients 9–14).

## `bench_undistort.sh` — throughput (optional)

Library A/B and per-element latency (library and element) at default and
pinned clocks, ~8 minutes. Needs `sudo` for `jetson_clocks`.
It expects a **second** library to A/B against (default
`../native/nvundistort/libnvundistort_manual.so`); pass your own as the first
argument. With only one library built, use `check_undistort.py` and skip this.

```bash
# run the Pi stream in a restart loop first (see the script header), then:
tests/bench_undistort.sh /path/to/other_lib.so
```

Output goes to `/tmp/bench_undistort/summary.txt`.
