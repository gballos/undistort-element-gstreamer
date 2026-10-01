# nvundistort

Lens undistortion for GStreamer on NVIDIA Jetson, on the GPU, without the frame
ever leaving GPU memory.

`nvundistort` is a GStreamer element that sits directly after
`nvv4l2decoder` and outputs undistorted NV12. It uses the same lens model as
OpenCV (`cv2.initUndistortRectifyMap`, rational radial + tangential) and
matches `cv2.remap` to within 0.02 grey levels on average. At 4608×2592 it
takes about 6 ms per frame on a Jetson Orin Nano.

```
nvv4l2decoder mjpeg=1 ! nvundistort params-file=/abs/path/params.txt
  ! 'video/x-raw(memory:NVMM),format=NV12' ! ...
```

The repository also holds an older variant of the same kernels as a plug-in
library for NVIDIA's `nvivafilter` element, and a calibration for the IMX708
sensor that you can use to try everything out.

- [Why an element](#why-an-element)
- [Requirements](#requirements)
- [Build](#build)
- [Quick start](#quick-start)
- [Using your own camera](#using-your-own-camera)
- [The element](#the-element)
- [Using it from an application](#using-it-from-an-application)
- [The nvivafilter library](#the-nvivafilter-library)
- [Tools](#tools)
- [Performance notes](#performance-notes)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)

## Why an element

On Jetson, `nvv4l2decoder mjpeg=1` outputs I420. Most of what follows wants
NV12, and at 4608×2592 that conversion alone costs about 20 ms in whichever
element does it (`nvvidconv`, or `nvivafilter` when it is linked to the
decoder). `nvundistort` reads the decoder's I420 surface through CUDA textures
and writes NV12 straight into its own output buffer, so the conversion happens
inside the remap and costs nothing extra.

Median time each element holds a frame, Jetson Orin Nano, 4608×2592 at
14 fps, measured with the GStreamer latency tracer:

| After `nvv4l2decoder` (about 20 ms) | Default clocks | Clocks pinned |
|---|---|---|
| `nvvidconv` + `nvivafilter` with the library | 21.3 + 22.7 ms | 7.4 + 11.7 ms |
| `nvundistort` | **6.3 ms** | **2.4 ms** |

"Clocks pinned" means `jetson_clocks` plus the VIC governor set to
`performance`; see [Performance notes](#performance-notes).
[docs/DATAFLOW.md](docs/DATAFLOW.md) shows both pipelines and the inside of
the element as diagrams, and [docs/DESIGN.md](docs/DESIGN.md) records the
measurements and what they showed about the platform.

## Requirements

Built and tested on one setup only:

- Jetson Orin Nano, JetPack 6.2 (Jetson Linux R36.4), CUDA 12.6
- DeepStream 7.1 (the element uses its NVMM buffer pool)
- GStreamer 1.20 with the development packages:
  `sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`
- For the Python tools: `python3` with `numpy` and `opencv-python`

Other Orin modules should work unchanged. For another Jetson family, pass its
CUDA architecture to make (`make ARCH=sm_72` for Xavier); that is untested.

## Build

On the Jetson:

```bash
make            # the element and the nvivafilter library
make element    # only the element   -> src/element/libgstnvundistort.so
make library    # only the library   -> src/libnvundistort.so
```

Nothing is installed system-wide. Point GStreamer at the element with
`GST_PLUGIN_PATH`:

```bash
export GST_PLUGIN_PATH=$PWD/src/element
gst-inspect-1.0 nvundistort
```

## Quick start

These use the bundled IMX708 calibration, which is for 4608×2592 frames. With
frames of another size the element warns and passes them through distorted;
see [Using your own camera](#using-your-own-camera).

```bash
export GST_PLUGIN_PATH=$PWD/src/element
PARAMS=$PWD/calibration/imx708_intrinsics_nvundistort.txt
```

**One JPEG in, one undistorted JPEG out:**

```bash
gst-launch-1.0 filesrc location=frame.jpg ! jpegparse ! nvv4l2decoder mjpeg=1 \
  ! nvundistort params-file=$PARAMS \
  ! 'video/x-raw(memory:NVMM),format=NV12' ! nvjpegenc ! filesink location=undistorted.jpg
```

**A live MJPEG stream, displayed:**

```bash
gst-launch-1.0 tcpclientsrc host=<camera-host> port=8881 ! jpegparse ! nvv4l2decoder mjpeg=1 \
  ! nvundistort params-file=$PARAMS stats-interval=50 \
  ! 'video/x-raw(memory:NVMM),format=NV12' ! nvvidconv ! autovideosink sync=false
```

A Raspberry Pi with an IMX708 camera can serve such a stream at the calibrated
resolution with:

```bash
rpicam-vid -t 0 --width 4608 --height 2592 --framerate 14 --codec mjpeg \
  --autofocus-mode manual --lens-position 0 --flush --listen -o tcp://0.0.0.0:8881
```

`--listen` serves one client and then exits, so wrap it in
`while true; do ...; sleep 1; done` if you reconnect often.

## Using your own camera

1. **Calibrate** the camera with OpenCV at the exact resolution and
   orientation you will stream, and save the result as an `.npz` with these
   keys:

   | Key | Content |
   |---|---|
   | `camera_matrix` | 3×3 matrix K |
   | `dist_coeffs` | up to 14 coefficients in OpenCV order (`k1 k2 p1 p2 k3 k4 k5 k6 ...`) |
   | `image_size` | `(width, height)` |
   | `model` | optional; `"fisheye"` is refused |

   For example, after `cv2.calibrateCamera` (optionally with
   `cv2.CALIB_RATIONAL_MODEL`):

   ```python
   rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None,
                                            flags=cv2.CALIB_RATIONAL_MODEL)
   np.savez("my_camera.npz", camera_matrix=K, dist_coeffs=dist.ravel(), image_size=(w, h))
   ```

2. **Write the parameter file.** The element cannot read an `.npz`; it reads a
   plain `key value` text file with one number per line:

   ```bash
   python3 tools/undistort_params.py my_camera.npz --alpha 0.0
   # -> my_camera_nvundistort.txt, next to the .npz (or use -o)
   ```

   | Keys | Taken from |
   |---|---|
   | `width`, `height` | `image_size` |
   | `fx`, `fy`, `cx`, `cy` | `camera_matrix` (K of the distorted input) |
   | `k1 k2 p1 p2 k3 k4 k5 k6` | `dist_coeffs` (missing ones are 0) |
   | `new_fx`, `new_fy`, `new_cx`, `new_cy` | `cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)`, the output camera matrix |

   `alpha` only affects the `new_*` values: 0 crops to valid pixels, 1 keeps
   the full field of view with black borders. The script refuses what the
   kernels cannot represent: the fisheye model, skew, and coefficients 9–14
   (thin prism and tilt).

3. **Pass it to the element** as `params-file`, with an absolute path.

A calibration only holds for the resolution, orientation and focus position it
was shot at.

### The bundled IMX708 calibration

`calibration/imx708_intrinsics.npz` and its parameter file
`calibration/imx708_intrinsics_nvundistort.txt` (alpha 0) come from a
checkerboard calibration of one IMX708 camera module: rational model, RMS
reprojection error 0.31 px, 4608×2592, no rotation, focus locked at lens
position 0. Lenses differ from unit to unit, so treat it as a working example
and a starting point, not as a calibration of your camera.

## The element

| | |
|---|---|
| Sink caps | `video/x-raw(memory:NVMM), format=I420`: what `nvv4l2decoder mjpeg=1` outputs |
| Src caps | `video/x-raw(memory:NVMM), format=NV12`, same size and colorimetry, pitch-linear |
| `params-file` | The parameter file. Read when the element starts. |
| `stats-interval` | N > 0: print the mean time per frame on stderr every N frames |

- **Pixel values are kept.** JPEG is full range, and the output surface is
  marked full range (`NV12_ER`), so later conversions treat it correctly.
- **Output buffers** come from the element's own pool of 4 to 8 NVMM buffers.
  A downstream element that holds on to more than 8 frames stalls the stream.
- **Outside the source frame** the output is black. With `alpha` 0 there are
  no such pixels.
- **If undistortion cannot run**, the element posts one GStreamer warning
  ending in `UNDISTORTION DISABLED, frames pass through distorted` and from
  then on only converts I420 to NV12, so the pipeline keeps running. Causes:
  no or unusable `params-file`, frames of another size than the calibration,
  an unexpected surface layout, any CUDA error.
- `GST_DEBUG=nvundistort:4` logs the layout of the first input and output
  surface.

Common errors:

- **`no element "nvundistort"`:** `GST_PLUGIN_PATH` does not include
  `src/element`, or the element is not built.
- **`could not link ... to nvundistort0`:** the upstream element does not
  output NVMM I420. Put the element directly after `nvv4l2decoder mjpeg=1`,
  with no `nvvidconv` in between.

## Using it from an application

Register the plugin directory, create the element like any other, and watch
the bus for warnings: that is how the element reports that it has stopped
undistorting.

```python
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)
Gst.Registry.get().scan_path("/abs/path/to/src/element")   # instead of GST_PLUGIN_PATH

pipeline = Gst.parse_launch(
    "tcpclientsrc host=<camera-host> port=8881 ! jpegparse ! nvv4l2decoder mjpeg=1 "
    "! nvundistort name=undistort params-file=/abs/path/params.txt "
    "! video/x-raw(memory:NVMM),format=NV12 ! fakesink")
pipeline.set_state(Gst.State.PLAYING)

bus = pipeline.get_bus()
while True:
    msg = bus.timed_pop_filtered(
        100 * Gst.MSECOND,
        Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING)
    if msg is None:
        continue
    if msg.type == Gst.MessageType.WARNING:
        print("warning:", msg.parse_warning()[0].message)   # e.g. UNDISTORTION DISABLED
        continue
    break
pipeline.set_state(Gst.State.NULL)
```

## The nvivafilter library

`src/libnvundistort.so` runs the same kernels inside NVIDIA's `nvivafilter`
element. It is slower, because `nvivafilter` copies every frame and the
library needs a second copy to remap in place, but it accepts NV12 as well as
I420 input.

```bash
export NVUNDISTORT_PARAMS=$PWD/calibration/imx708_intrinsics_nvundistort.txt
gst-launch-1.0 tcpclientsrc host=<camera-host> port=8881 ! jpegparse ! nvv4l2decoder mjpeg=1 \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvivafilter cuda-process=true customer-lib-name=$PWD/src/libnvundistort.so \
  ! 'video/x-raw(memory:NVMM),format=NV12' ! nvvidconv ! autovideosink sync=false
```

| Variable | Meaning |
|---|---|
| `NVUNDISTORT_PARAMS` | Parameter file (required) |
| `NVUNDISTORT_MODE` | `off` (return immediately), `map` (map the surface only), `y` (luma only), `full` (default). The first three are bring-up aids that separate the costs. |
| `NVUNDISTORT_STATS` | Print the mean time per frame every N frames |

Things to know:

- `nvivafilter` compresses luma to 16–235 and marks its output limited range.
  A later conversion expands it again, at the cost of some rounding.
- Any CUDA error disables undistortion for the rest of the run. It is printed
  once on stderr as `[nvundistort] ... UNDISTORTION DISABLED`, and frames then
  pass through distorted.
- The library builds against NVIDIA's `customer_functions.h`
  (`src/include/`), the `nvivafilter` plug-in interface. If your Jetson Linux
  release ships a different one, take it from `nvsample_cudaprocess_src.tbz2`
  in that release's "Driver Package (BSP) Sources" and replace the file.

## Tools

Run these on the Jetson, from the repository root, after `make`.

**Accuracy: `tools/check_undistort.py`.** Runs one JPEG through the
undistortion and compares the result with `cv2.remap`. PASS means a mean
absolute difference below 1 level on Y, U and V. It also writes
`undistort_preview.jpg`: straight edges should be straight.

```bash
python3 tools/check_undistort.py frame.jpg --npz calibration/imx708_intrinsics.npz --element --out-dir /tmp   # element
python3 tools/check_undistort.py frame.jpg --npz calibration/imx708_intrinsics.npz --out-dir /tmp             # library
```

The frame must be shot at the calibrated resolution and orientation. Typical
result for the element: Y mean 0.02, U 0.09, V 0.07 levels. The largest
differences are in the outermost pixel ring, where OpenCV blends in the border
colour and the kernels do not. If `import cv2` fails with a NumPy 2 error, a
user-installed NumPy is shadowing the one OpenCV was built with: run with
`PYTHONNOUSERSITE=1`.

**Parameter file: `tools/undistort_params.py`.** See
[Using your own camera](#using-your-own-camera).

**Benchmark: `tools/bench_undistort.sh`.** On a live MJPEG stream, measures
the time per frame of the library and the time each element holds a frame,
with no undistortion, with the library and with the element, at default
clocks and with `jetson_clocks`. Takes about 6 minutes and needs `sudo`. The
stream address and other settings are environment variables listed at the top
of the script. Results go to `/tmp/bench_undistort/summary.txt`.

```bash
HOST=<camera-host> PORT=8881 tools/bench_undistort.sh
```

## Performance notes

- **Clock scaling matters more than anything else.** The work is a short burst
  every frame, so at default clocks the GPU and the VIC (the Jetson's video
  converter) sit at low frequencies most of the time.
  - `sudo jetson_clocks` pins the GPU. The element went from 6.3 ms to 2.4 ms.
  - `jetson_clocks` does **not** pin the VIC. Its governor is set separately:

    ```bash
    echo performance | sudo tee /sys/devices/platform/bus@0/13e00000.host1x/15340000.vic/devfreq/15340000.vic/governor
    ```

    Both settings last until reboot and cost idle power and heat.
- **Removing `nvvidconv` can slow down later VIC stages at default clocks.**
  With less VIC work per frame, the VIC's governor lowers its clock, and
  anything downstream that uses the VIC (`nvvidconv`, `nvvideoconvert`,
  `nvstreammux` scaling) takes longer. Pinning the VIC removes the effect.
- **Mapping a buffer into CUDA costs about 3 ms**, so the element maps each
  buffer once and keeps the mapping. The decoder cycles through four buffers.

## Limitations

- **Jetson only.** The element relies on NVMM surfaces and CUDA–EGL interop.
- **I420 input only.** That is what `nvv4l2decoder mjpeg=1` outputs. For
  another source you can put `nvvidconv ! 'video/x-raw(memory:NVMM),format=I420'`
  in front; it links and undistorts, but in a test that conversion compressed
  luma to 0–235, and it costs time. The `nvivafilter` library takes NV12
  directly.
- **8-bit, even width and height.**
- **Lens model:** OpenCV's pinhole model with rational radial and tangential
  distortion. No fisheye model, no skew, no thin-prism or tilt terms.
- **One fixed frame size**, the one in the parameter file. The parameter file
  is read at start; changing it needs a restart.
- **GPU 0, one stream per element instance.**
- **Tested** at 4608×2592 on an Orin Nano, for runs of a few minutes. It has
  not had a long soak test.

## Repository layout

```
Makefile                 builds src/
src/
  undistort_kernels.cuh  the lens model and the remap kernels (shared)
  undistort_params.h     parameter file reader (shared)
  element/               the nvundistort GStreamer element
    gstnvundistort.cpp     GStreamer side: caps, output pool, properties
    undistort_engine.cu    CUDA side: surface mapping, kernel launches
  nvundistort.cu         the nvivafilter library
  include/               NVIDIA's customer_functions.h
tools/                   parameter file writer, accuracy check, benchmark
calibration/             IMX708 calibration and its parameter file
docs/                    dataflow diagrams and design notes
```

## License

[MIT](LICENSE). `src/include/customer_functions.h` is NVIDIA's, under the BSD
3-clause licence in its header.
