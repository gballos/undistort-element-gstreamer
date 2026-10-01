# nvundistort

Exact lens undistortion of the camera stream on the Jetson's GPU, in two forms
that share the same maths and kernels (`undistort_kernels.cuh`) and the same
[parameter file](#parameter-file):

- **The `nvundistort` GStreamer element** (`element/`), which takes the
  decoder's frames directly. This is the faster one: see
  [The nvundistort element](#the-nvundistort-element).
- **The `nvivafilter` library** (`libnvundistort.so`), described first below.
  It is what `pipeline/builder.py` uses today.

[DATAFLOW.md](DATAFLOW.md) shows both pipelines and the inside of the element
as diagrams.

## The nvivafilter library

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

Neither the library nor the element can read a `.npz`, so the calibration is turned into a plain
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

## Build the library (on the Jetson)

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

## Library bring-up, in order

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

## When something is wrong (library)

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

## The nvundistort element

`nvv4l2decoder mjpeg=1` outputs I420, and converting that to NV12 at 4608x2592
costs about 22 ms in whichever element does it (`nvvidconv`, or `nvivafilter`
when it is linked to the decoder directly). The element avoids that step: its
kernels read the decoder's I420 surface through textures and write NV12
straight into a buffer from the element's own pool, so the conversion happens
inside the remap. No `nvvidconv` in front, no `nvivafilter`, no scratch copy.

```
nvv4l2decoder mjpeg=1 ! nvundistort params-file=<abs path>
  ! video/x-raw(memory:NVMM),format=NV12 ! tee ...
```

Median time per element at default clocks, 4608x2592 at 14 fps (GStreamer
latency tracer, saved frame looped):

| After the decoder (19.9 ms) | Added | Decode -> undistorted frame |
|---|---|---|
| `nvvidconv` + `nvivafilter` library | 21.3 + 22.7 ms | 63.6 ms |
| `nvivafilter` library linked to the decoder | 32.1 ms | 51.8 ms |
| `nvundistort` element | 6.3 ms | 26.2 ms |

| | |
|---|---|
| Sink caps | `video/x-raw(memory:NVMM), format=I420` (it does not link to anything else) |
| Src caps | `video/x-raw(memory:NVMM), format=NV12`, same size and colorimetry |
| `params-file` | The [parameter file](#parameter-file). Read when the element starts. |
| `stats-interval` | N > 0: print the mean time per frame on stderr every N frames |

The output keeps the decoder's pixel values: full range for JPEG, and the
output surface is marked full range (`NV12_ER`), so downstream conversions
treat it correctly. `nvivafilter` instead compresses luma to 16-235.

Files: `element/gstnvundistort.cpp` is the GStreamer side (caps, output pool,
properties), `element/undistort_engine.cu` the CUDA side. They are separate
because nvcc and the glib headers do not mix.

### Build (on the Jetson)

Needs DeepStream (for its NVMM buffer pool) and the GStreamer development
packages. `customer_functions.h` is not needed.

```bash
make -C native/nvundistort/element
export GST_PLUGIN_PATH=$PWD/native/nvundistort/element
gst-inspect-1.0 nvundistort
```

### Bring-up

**1. Accuracy**, against `cv2.remap` of the plain decode:

```bash
cd tests && python3 check_undistort.py frame.jpg --npz ../imx708_intrinsics.npz --element --out-dir /tmp
```

**2. Throughput** on the live stream:

```bash
gst-launch-1.0 -v tcpclientsrc host=192.168.10.1 port=8881 ! jpegparse ! nvv4l2decoder mjpeg=1 \
  ! nvundistort params-file=$PWD/imx708_intrinsics_nvundistort.txt stats-interval=50 \
  ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false
```

To watch it, swap the `fpsdisplaysink ...` sink for `nvvidconv ! autovideosink sync=false`.

### When something is wrong (element)

- **Undistortion cannot run:** the element posts one GStreamer warning ending
  in `UNDISTORTION DISABLED, frames pass through distorted`, and from then on
  only converts I420 to NV12, so the pipeline keeps running. Causes: no or
  unusable `params-file`, frames of another size than the calibration, an
  unexpected surface layout, any CUDA error.
- **`could not link ... to nvundistort0`:** the upstream element does not
  output NVMM I420. The element goes directly after `nvv4l2decoder`, with no
  `nvvidconv` in between.
- **`no element "nvundistort"`:** `GST_PLUGIN_PATH` does not include
  `native/nvundistort/element`.
- `GST_DEBUG=nvundistort:4` logs the layout of the first input and output
  surface.
