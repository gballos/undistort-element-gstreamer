# Design notes and measurements

Why the element is built the way it is, what was measured to get there, and
what is still untested. For diagrams see [DATAFLOW.md](DATAFLOW.md); for usage
see the [README](../README.md).

## Setup for every measurement

Jetson Orin Nano (JetPack 6.2, DeepStream 7.1, 15 W power mode), IMX708 camera
stream as MJPEG at 4608×2592 / 14 fps. Times are the median each element holds
a frame, from the GStreamer latency tracer
(`GST_TRACERS='latency(flags=element)'`). Unless a row says "live", the source
is one saved JPEG looped at 14 fps:

```
multifilesrc location=frame.jpg loop=true caps=image/jpeg,framerate=14/1
  ! jpegparse ! identity sync=true ! nvv4l2decoder mjpeg=1 ! ...
```

## Results

### The undistortion stage alone

Default clocks:

| After `nvv4l2decoder` (19.9 ms) | `nvvidconv` | `nvivafilter` | `nvundistort` | Decode → output |
|---|---|---|---|---|
| nothing | – | – | – | 20.1 ms |
| `nvvidconv` | 22.1 | – | – | 41.8 ms |
| `nvvidconv` + `nvivafilter` library | 21.3 | 22.7 | – | 63.6 ms |
| `nvivafilter` library linked to the decoder | – | 32.1 | – | 51.8 ms |
| `nvundistort` element | – | – | 6.3 | 26.2 ms |

On a live stream the element measured the same: 6.4 ms, 14 fps, no dropped
frames.

### With work downstream, and the effect of clocks

The same trunks feeding a `tee` with two consumers that use the VIC:
`nvstreammux` scaling to 960×544, and `nvvideoconvert` to RGBA in system
memory. Decoder about 20 ms in every row.

| Trunk | Clocks | trunk | `nvstreammux` | RGBA convert | VIC clock, mean |
|---|---|---|---|---|---|
| `nvvidconv` | default | 15.3 | 9.8 | 37.7 | 293 MHz |
| `nvundistort` | default | 6.4 | 19.1 | 51.1 | 205 MHz |
| `nvvidconv` | GPU pinned | 19.9 | 9.2 | 28.6 | 274 MHz |
| `nvundistort` | GPU pinned | 2.4 | 18.9 | 42.3 | 188 MHz |
| `nvvidconv` | GPU + VIC pinned | 7.4 | 6.0 | 24.4 | 435 MHz |
| `nvundistort` | GPU + VIC pinned | 2.4 | 6.1 | 24.9 | 435 MHz |
| `nvvidconv` + `nvivafilter` library | GPU + VIC pinned | 7.4 + 11.7 | 6.0 | 24.4 | 435 MHz |

"GPU pinned" is `sudo jetson_clocks`. "VIC pinned" is the VIC devfreq
governor set to `performance` (see the README's performance notes).

### Accuracy

Against `cv2.remap` with the maps from `cv2.initUndistortRectifyMap`:

- Element, reference = plain decode downloaded through `nvvidconv`: mean
  absolute difference 0.017 (Y), 0.090 (U), 0.066 (V) grey levels.
- Element, reference = the exact planes CUDA reads: chroma mean 0.010, interior
  maximum 3 levels.
- Library (`nvivafilter`), reference = its own pass-through: 0.034 (Y),
  0.018 (U), 0.019 (V).
- Six different frames looped twice each matched their own frame, so a kept
  mapping does not serve stale data.

The few large differences are in the outermost pixel ring, where OpenCV blends
in the border colour and the kernels do not.

### Stability

- 2.5 minutes feeding the `tee` pipeline above: no warnings, process memory
  flat (547 → 551 MB).
- Nine start/stop cycles (six to end of stream, three interrupted): clean
  exits, available system memory unchanged.
- 35 s on a live stream with a detector and tracker running on the same GPU:
  clean, 13.3 and 13.5 frames per second on the two branches.

## What the measurements showed about the platform

- **`nvv4l2decoder mjpeg=1` outputs I420**, not NV12: pitch-linear, three
  planes, full range (`colorimetry=1:4:5:1`, surface format `YUV420_ER`), from
  a pool of four buffers that recycle.
- **The I420 → NV12 conversion is the expensive step**, about 22 ms at default
  clocks. Whichever element sees I420 first pays it: `nvvidconv`, or
  `nvivafilter` when linked to the decoder (22 ms doing nothing, against
  12.8 ms when it is fed NV12).
- **`nvivafilter` compresses luma to 16–235** (linear fit slope 0.868,
  residual 0.6 levels) and marks its output limited range. A later conversion
  to RGBA expands it again, so the levels come out the same, minus rounding.
  `nvvidconv` to NVMM NV12 keeps full range.
- **CUDA can read the decoder's surface directly.** Mapped through EGL it is
  pitch memory, 512-byte aligned, and textures work on it. No copy is needed.
- **Mapping costs about 3.3 ms per frame** (register plus unregister). With
  the mapping kept per buffer, the whole undistortion is about 6 ms at default
  clocks.
- **The VIC has its own clock governor, and `jetson_clocks` does not pin it.**
  With less VIC work per frame the governor lowers the clock, which is why the
  stages after the element got slower at default clocks in the table above.
  With the VIC pinned they take the same time whichever trunk feeds them.
- **The UV plane pitch of NV12 is reported in U,V pairs** (2304 for 4608-byte
  rows) by the CUDA–EGL frame description.
- **`nvdewarper` is not a substitute:** it targets 360° and fisheye cameras.
- **`nvdsvideotemplate` is not a shortcut:** it is a thin shell around a
  custom library that must own the output pool and push buffers from its own
  thread. A plain `GstBaseTransform` is less code.

## Design decisions

- **A `GstBaseTransform` with its own output pool.** Input and output are
  different buffers, so the remap needs no scratch copy. The pool is
  DeepStream's `gst_nvds_buffer_pool` with 4 to 8 buffers; each is a full
  frame (18 MB at 4608×2592), hence the cap.
- **I420 in, NV12 out.** Luma is remapped by `remap_y`; `remap_uv_planar`
  reads U and V from two one-channel textures and writes interleaved pairs.
  The conversion is free because the remap writes every output byte anyway.
- **No lookup table.** The source position is computed per pixel with the
  model of `cv2.initUndistortRectifyMap` (unproject with the new camera
  matrix, apply the lens model, project with K). Texture fetches use `+0.5`
  because texel centres sit at half-integer coordinates. The texture unit
  blends with 8-bit weights, so a result can differ from a float blend by one
  level.
- **Mappings kept per buffer**, keyed by dmabuf fd. Each mapping holds its own
  copy of the surface description, so unmapping never touches a buffer that
  may already be freed. An fd that comes back with a different surface is
  remapped, more than 16 mappings resets the cache, and all mappings are
  dropped on a caps change and on stop.
- **Fail soft.** Any failure posts one warning and switches to a plain
  `NvBufSurfTransform` I420 → NV12 conversion, so the stream keeps running.
  CUDA errors are sticky, so the engine is not retried.
- **Two translation units.** nvcc defines `__noinline__` as a macro, which
  breaks the glib headers, so the GStreamer side (`gstnvundistort.cpp`, g++)
  and the CUDA side (`undistort_engine.cu`, nvcc) are compiled separately.
- **Output marked full range.** The output caps carry the decoder's
  colorimetry, and the pool then allocates `NV12_ER` surfaces.
- **Parameter file parsed in the C locale.** `gst-launch` applies the system
  locale, and under a comma-decimal one `%lf` stops at the `.`.

## Not tested

- A soak test of hours, and long runs next to heavy GPU inference.
- The fallback conversion while other threads also use `NvBufSurfTransform`
  heavily.
- Resolutions other than 4608×2592, other Jetson modules, other JetPack and
  DeepStream versions.
- Formats other than what `nvv4l2decoder mjpeg=1` produces from 4:2:0 JPEG.

## Developing without a Jetson

Compile checks of the `.cu` files work on an x86 machine without a GPU, using
NVIDIA's CUDA redistributables (`cuda_nvcc` and `cuda_cudart` archives from
`https://developer.download.nvidia.com/compute/cuda/redist/`, unpacked into
one directory) and `nvcc -arch=sm_87`. To check that a change to the kernels
alters nothing, compare the `-ptx` output before and after.
