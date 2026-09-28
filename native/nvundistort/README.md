# nvundistort

`libnvundistort.so` is a plug-in library for NVIDIA's `nvivafilter` GStreamer element.
It undistorts every NV12 frame in place on the GPU, using the exact OpenCV
model (rational radial + tangential) from a `utils/calibration/calibrate.py` `.npz`.
`pipeline/builder.py` inserts it before the tee when the active camera profile
sets `calibration_npz` in `config.yaml`:

```
nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12
  ! nvivafilter cuda-process=true customer-lib-name=.../libnvundistort.so
  ! video/x-raw(memory:NVMM),format=NV12 ! tee ...
```

nvivafilter does not negotiate with `nvv4l2decoder` directly, so an `nvvidconv`
sits in between.

For each frame, `gpu_process()` maps the surface into CUDA, copies Y and UV to
scratch buffers, and runs one remap kernel per plane. The source position of
each output pixel is computed in the kernel, the same way as
`cv2.initUndistortRectifyMap`, so there is no lookup table. The kernels read
the scratch copies through CUDA textures, so the texture hardware does the
bilinear blend; its 8-bit weights can put a pixel 1–2 levels away from OpenCV.
The maths and kernels are in `undistort_kernels.cuh`, the `nvivafilter`
wrapper in `nvundistort.cu`.

## Parameter file

The library cannot read a `.npz`, so the calibration is turned into a plain
`key value` text file first, one number per line. For the committed
`imx708_intrinsics.npz` that file is `imx708_intrinsics_nvundistort.txt`, also
committed. It is made from the `.npz` by `utils/undistort_params.py`:

```bash
python3 -m utils.undistort_params imx708_intrinsics.npz --alpha 0.0
```

| Keys | Taken from |
|---|---|
| `width`, `height` | `image_size` |
| `fx`, `fy`, `cx`, `cy` | `camera_matrix` (K of the distorted input) |
| `k1 k2 p1 p2 k3 k4 k5 k6` | `dist_coeffs` (OpenCV order; missing ones are 0) |
| `new_fx`, `new_fy`, `new_cx`, `new_cy` | `cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)`, the output camera matrix |

`alpha` only affects the `new_*` values: 0 crops to valid pixels, 1 keeps the
full field of view with black borders. The script refuses what the kernel
cannot represent (fisheye model, skew, coefficients 9-14).

`pipeline/builder.py` rewrites this file from `config.yaml`'s `undistort.alpha`
every time it builds the pipeline, so the running pipeline never uses a stale
one. The committed copy is for running the library by hand (bring-up below)
and matches `alpha: 0.0`. After a new calibration or a change of alpha,
regenerate it with the command above and commit it together with the `.npz`.

## Build (on the Jetson)

1. Get NVIDIA's `customer_functions.h`, which defines the plug-in interface. It
   is not vendored here, so the library always builds against the interface
   the installed `nvivafilter` was built with. Check whether it is on the device:

   ```bash
   find / -name customer_functions.h 2>/dev/null
   ```

   If it isn't, it ships in the Jetson Linux sources that match your JetPack
   (JetPack 6.2 = Jetson Linux R36.4.x). On the release page, download
   "Driver Package (BSP) Sources" (`public_sources.tbz2`), then:

   ```bash
   tar xjf public_sources.tbz2 Linux_for_Tegra/source/nvsample_cudaprocess_src.tbz2
   tar xjf Linux_for_Tegra/source/nvsample_cudaprocess_src.tbz2
   ```

   Copy `customer_functions.h` into `native/nvundistort/include/`.

2. Build:

   ```bash
   make -C native/nvundistort
   ```

## Bring-up, in order

Run these from the repo root on the Jetson. Each step has a clear pass/fail.

**1. Accuracy.** Pass one frame shot at the calibrated resolution and
orientation through the filter, and compare it with `cv2.remap`:

```bash
python3 -m tools.check_undistort frame.jpg --npz imx708_intrinsics.npz --out-dir /tmp
```

PASS means a mean absolute difference below 1 level on Y, U and V. Also look
at `/tmp/undistort_preview.jpg`: straight edges should be straight.

**2. Throughput.** Compare the live stream with and without the filter. The
modes separate the costs: `off` is nvivafilter alone, `map` adds the CUDA
mapping of each frame, and `full` is the complete undistortion.

```bash
export NVUNDISTORT_PARAMS=$PWD/imx708_intrinsics_nvundistort.txt   # committed, alpha=0
export NVUNDISTORT_STATS=50           # print mean ms/frame every 50 frames
LIB=$PWD/native/nvundistort/libnvundistort.so

# baseline, no filter
gst-launch-1.0 -v tcpclientsrc host=192.168.10.1 port=8881 ! jpegparse ! nvv4l2decoder mjpeg=1 \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false

for m in off map full; do
  NVUNDISTORT_MODE=$m gst-launch-1.0 -v tcpclientsrc host=192.168.10.1 port=8881 ! jpegparse \
    ! nvv4l2decoder mjpeg=1 ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvivafilter cuda-process=true customer-lib-name=$LIB \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false
done
```

To watch it, swap the `fpsdisplaysink ...` sink for `nvvidconv ! autovideosink sync=false`.

**3. Pipeline.** Set `camera.active: imx708` (its `calibration_npz` is already
set) and run the app as usual. At startup `python -m config` prints
`undistort: nvivafilter ...`, and the library prints its own `[nvundistort]` line.

## Environment

| Variable | Meaning |
|---|---|
| `NVUNDISTORT_PARAMS` | Parameter file (required; see [Parameter file](#parameter-file)). `builder.py` sets it itself. |
| `NVUNDISTORT_MODE` | `off` \| `map` \| `y` (luma only) \| `full` (default) |
| `NVUNDISTORT_STATS` | Print the mean `gpu_process` time every N frames |

## When something is wrong

- **Any CUDA error disables undistortion for the rest of the run.** The error
  is printed once as `[nvundistort] ... UNDISTORTION DISABLED`, and frames then
  pass through distorted. Check stderr after a run.
- **Sheared image:** row pitch is being mixed up with width.
- **Correct shape, wrong colours:** UV plane indexing. `NVUNDISTORT_MODE=y`
  isolates it.
- **`unexpected surface` at startup:** nvivafilter handed over something other
  than a pitch-linear NV12 frame at the calibrated size.
- **Compile error about `init` linkage or `CustomerFunction` members:** the
  header differs from the one this was written against. Paste the error.
