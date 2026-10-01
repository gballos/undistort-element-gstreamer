# Plan: replace nvivafilter with our own undistortion element

Hand-off document. It is self-contained: what was measured, what is built, what
is left, and the working conventions.

**Status (2026-10-01):** the element exists (`element/`), builds on the Jetson
and passes the accuracy check and a short stability run. It is **not yet wired
into the application** (`pipeline/builder.py` in `edge-inference`); steps 4-6
below are open.

---

## 1. Why

Measured on the Jetson Orin Nano, IMX708 at 4608×2592 / 14 fps, camera stream
only (no inference running). The GStreamer latency tracer gives the median
time each element holds a frame. Default clocks, saved frame looped at 14 fps:

| Pipeline after `nvv4l2decoder` (19.9 ms) | `nvvidconv` | `nvivafilter` | `nvundistort` | Decode → output |
|---|---|---|---|---|
| nothing | – | – | – | 20.1 ms |
| `nvvidconv` (undistortion off) | 22.1 | – | – | 41.8 ms |
| `nvvidconv` + `nvivafilter` library (**today**) | 21.3 | 22.7 | – | **63.6 ms** |
| `nvivafilter` library linked to the decoder | – | 32.1 | – | 51.8 ms |
| **`nvundistort` element** | – | – | 6.3 | **26.2 ms** |

On the live stream the element measured the same (6.4 ms; 14 fps, no drops).

- **The element saves about 37 ms per frame** against today's pipeline, and is
  about 15 ms faster than today's pipeline with undistortion *off*.
- **In context:** the documented glass-to-glass budget is ~214 ms, of which the
  Pi's capture + JPEG encode is ~164 ms (wiki page *Gaze-Pipeline-Performance*,
  measured on the older IMX219). That budget omits the `nvvidconv` and
  undistortion stages.
- Pinned clocks (`jetson_clocks`) have **not** been measured for the element.
  For the library, pinning took its own time from ~10 ms to ~4 ms and left the
  video hardware stages unchanged (§8).

## 2. What the measurements showed

These replace the assumptions of the first version of this plan.

- **`nvv4l2decoder mjpeg=1` outputs I420**, not NV12: pitch-linear, three
  planes, full range (`colorimetry=1:4:5:1`, surface format `YUV420_ER`), from
  a pool of four buffers that recycle. It is not block-linear.
- **The ~22 ms is the I420 → NV12 conversion.** Whichever element sees I420
  first pays it: `nvvidconv` (22 ms), or `nvivafilter` when linked to the
  decoder directly (22 ms doing nothing, against 12.8 ms when it is fed NV12).
  So dropping `nvvidconv` in front of `nvivafilter` saves only ~12 ms.
- **`nvivafilter` does link to the decoder directly** (its sink accepts I420).
- **`nvivafilter` compresses luma to 16–235** (linear fit slope 0.868,
  residual 0.6 levels). This is the "softening" seen earlier; it is not blur.
  Its output is marked limited range, so a later conversion to RGBA expands it
  again and the levels come out the same, minus the rounding of the round
  trip. `nvvidconv` to NVMM NV12 keeps full range.
- **CUDA can read the decoder's surface directly.** Mapped through EGL it is
  pitch memory, 512-byte aligned, and textures work on it. No scratch copy.
- **Mapping costs ~3.3 ms per frame** (register + unregister), so it is kept
  per buffer. With the mapping kept, the whole undistortion is ~6.1 ms.
- **`nvdsvideotemplate` is not a shortcut.** It is a thin shell: the custom
  library must own the output pool and push buffers from its own thread
  (NVIDIA's sample is 1050 lines). A plain `GstBaseTransform` is simpler.
- **`nvdewarper`** was tried earlier and abandoned: it's for 360°/fisheye
  cameras, gave wrong geometry, and was slow.

Other facts worth keeping:
- The Jetson reports NV12's UV plane pitch in **U,V pairs** (2304 for
  4608-byte rows).
- `gst-launch` applies the system locale (comma decimal). `sscanf` and `awk`
  must parse in the C locale (`uselocale`, `LC_ALL=C`).
- Texture fetches need `+0.5` on coordinates (texel centres).
- nvcc defines `__noinline__` as a macro, which breaks the glib headers. Keep
  GStreamer code in a `.cpp` compiled by g++.
- The `nvvidconv` that downloads a frame to system memory filters chroma on
  sharp colour edges (up to ~33 levels on a few thousand pixels). It affects
  the *reference* of the accuracy check, not the element.
- The calibration is valid only for **4608×2592, no rotation, focus locked**:
  the Pi runs `rpicam-vid -t 0 --width 4608 --height 2592 --framerate 14
  --codec mjpeg --autofocus-mode manual --lens-position 0 --flush --listen -o
  tcp://192.168.10.1:8881`.
  `--listen` serves one client and exits, so for repeated tests wrap it in
  `while true; do ...; sleep 1; done`.

## 3. The element

```
nvv4l2decoder mjpeg=1 ! nvundistort params-file=<abs path>
  ! video/x-raw(memory:NVMM),format=NV12 ! tee ...
```

| File | Role |
|---|---|
| `undistort_kernels.cuh` | `Params`, `src_coords` (exact OpenCV rational + tangential model), `make_plane_texture`, kernels `remap_y`, `remap_uv` (NV12 source, for the library) and `remap_uv_planar` (I420 source, for the element). |
| `undistort_params.h` | `load_params`: reads the `key value` parameter file. Shared by the library and the element. |
| `element/gstnvundistort.cpp` | The GStreamer side, a `GstBaseTransform`: caps (I420 in, NV12 out), output pool (DeepStream's `gst_nvds_buffer_pool`, 4–8 buffers), properties `params-file` and `stats-interval`, the fallback conversion. |
| `element/undistort_engine.cu`, `.h` | The CUDA side: maps each surface once and keeps the mapping by dmabuf fd, launches the two kernels, waits for them. |
| `element/Makefile` | Builds `libgstnvundistort.so` on the Jetson. Use with `GST_PLUGIN_PATH`. |
| `nvundistort.cu` | The `nvivafilter` library, unchanged in behaviour (its output is byte-identical after `load_params` moved out). |

Behaviour:
- **Output** is NV12 of the same size, pitch-linear, marked full range
  (`NV12_ER`) because the caps carry the decoder's colorimetry.
- **Failure** (no or unusable `params-file`, another frame size, unexpected
  surface, any CUDA error): one GStreamer warning ending in `UNDISTORTION
  DISABLED`, then the element only converts I420 → NV12 with
  `NvBufSurfTransform`, so frames keep flowing, distorted.
- **Kept mappings** are dropped on a caps change and on stop. An fd that comes
  back with a different surface is remapped; more than 16 mappings resets the
  cache.
- Undistortion must stay **before the tee**: the assembler scales detection
  boxes onto the gaze frame, so both branches need the same geometry.

## 4. Steps

**Done**

- **Step 0 — Probe.** Sources and caps read on the device (§2).
- **Step 1 — Spike.** A standalone program proved the direct read and the 6 ms.
- **Step 2 — The element.** Built as described in §3.
- **Step 3 — Accuracy.** `tests/check_undistort.py --element`, reference =
  plain decode: Y mean 0.017, U 0.090, V 0.066 levels, PASS. Against the exact
  planes CUDA reads, chroma is 0.010 mean with an interior maximum of 3 levels.
  Six different frames looped twice each matched their own frame, so a kept
  mapping does not serve stale data.
- **Stability, short.** 2.5 minutes in `tee` → `nvstreammux` (960×544) + `tee`
  → `nvvidconv` → RGBA: no warnings, process memory flat (547 → 551 MB). Nine
  start/stop cycles (six to EOS, three interrupted) left available memory
  unchanged.

**Open**

- **Step 4 — Builder.** In `edge-inference`, `pipeline/builder.py`
  `_insert_undistort` links `dec → nvundistort → caps` instead of
  `nvvidconv → caps → nvivafilter → caps`, sets `params-file` instead of
  `NVUNDISTORT_PARAMS`, and the process needs `GST_PLUGIN_PATH` (or
  `Gst.Registry.scan_path`) for `native/nvundistort/element`. Copy `element/`,
  `undistort_params.h` and the updated `undistort_kernels.cuh`,
  `nvundistort.cu` and `Makefile` into that repo. Keep `nvivafilter` as a
  fallback until step 5 passes, then remove it in one commit.
- **Step 5 — Evaluate in the application.** `tests/bench_undistort.sh` (it now
  includes the element; needs `sudo`), then the tracer on `python -m apps.run`
  and a 30-minute soak with `tegrastats`. *Pass:* decode → undistorted frame
  ≈ 26 ms at default clocks, no `UNDISTORTION DISABLED`, flat memory, no CUDA
  errors with inference running.
- **Step 6 — Docs.** Update the wiki pages (§7).

**Risks:** the element adds GPU work before the tee while inference runs; the
builder notes crashes when both tee branches shared a transform engine
(`DET_COMPUTE_HW` comment), so the soak in step 5 matters. The output pool is
capped at 8 buffers of 18 MB; a downstream element that holds more than that
would stall the stream. The DeepStream pool API may change on upgrade.

## 5. Testing on the device

The Jetson is reachable over SSH from the dev machine (`jetson@192.168.1.210`).
`edge-inference` lives in `~/edge-inference`; this repo was copied to
`~/undistort-element-gstreamer` and built there. `sudo` needs a password, so
`jetson_clocks` runs are the user's.

```bash
make -C native/nvundistort/element
export GST_PLUGIN_PATH=$PWD/native/nvundistort/element
cd tests && PYTHONNOUSERSITE=1 python3 check_undistort.py /tmp/frame_0001.jpg \
  --npz ../imx708_intrinsics.npz --element --out-dir /tmp
```

`PYTHONNOUSERSITE=1` is needed on this Jetson because a user-installed NumPy 2
shadows the one the system OpenCV was built with.

Without the Pi, a saved JPEG can stand in for the stream:
`multifilesrc location=frame.jpg loop=true caps=image/jpeg,framerate=14/1 !
jpegparse ! identity sync=true ! nvv4l2decoder mjpeg=1 ! ...`.

Compile checks of the `.cu` files also work on a dev machine without a GPU,
with NVIDIA's CUDA 12.6 redistributables (`cuda_nvcc` and `cuda_cudart`
archives from `https://developer.download.nvidia.com/compute/cuda/redist/`,
unpacked into one directory) and `nvcc -arch=sm_87`. To prove a refactor of
the kernels changes nothing, compare the `-ptx` output before and after.

## 6. Still open (independent of the element)

- **App-level measurements** (see step 5). **Note:** the `utils/latency` suite
  cannot see undistortion. Its round-trip runs bypass DeepStream, and its stage
  timing starts after the appsink.
- **Clock policy for production:** see §8. Decide whether the deployed Jetson
  runs `jetson_clocks` at boot, and record the `nvpmodel` mode (15 W today).
- **`utils/csi_camera.py` (gaze-calibration app) does not undistort,** so
  gaze-to-screen calibrations recorded with it don't match production
  geometry.
- **Without undistortion the pipeline still pays ~22 ms** in `nvvidconv` for
  I420 → NV12. `nvstreammux` lists I420 on its sink, so it may be removable
  there too. Needs its own test.
- **IMX219 orientation:** its old docs rotated on the Pi *and* flipped on the
  Jetson; now it does neither. Check before switching back to it.
- **Calibration focus:** confirm the calibration frames were shot with the
  focus locked (`--lens-position 0`).

## 7. Documentation locations

- Wiki (git repo, Azure DevOps format, `%2D` = `-` in file names, `.order`
  files list page order): `~/DOOH/docs/ELIAS---AdaptiveSight.wiki/`.
  - `Infrastructure/Camera-Stream-%2D-Infrastructure-and-Consumption-Pipeline.md` (AS-BUILT)
  - `.../Camera-Stream-.../pipeline%2Ddiagram.md` (mermaid diagram + GPU/CPU table)
  - `.../Camera-Stream-.../Lens%2DUndistortion.md` (process, performance, inefficiencies, limitations)
- Library and element docs: `native/nvundistort/README.md`.

## 8. Pinned vs default clocks, in short

`jetson_clocks` locks the CPU, GPU and memory controller at their maximum
frequency within the current power mode (`nvpmodel`), instead of letting them
scale with load. The undistortion does a short burst of work every 71 ms, and
at default clocks the GPU stays mostly at low frequency. That's why the
library took ~10 ms at default clocks and ~4 ms pinned. The video hardware
(decoder, VIC in `nvvidconv`, `nvivafilter`'s copy) barely changes. Under full
inference load the GPU clocks up by itself, so production likely sits between
the two. Pinning at boot costs idle power and heat. It isn't persistent unless
started by a systemd unit.

## 9. Working conventions

- Make the smallest change that does the job. Match the surrounding code's
  style and comment density.
- Device commands can be run over SSH. Ask the user before anything
  destructive on the Jetson, and leave `~/edge-inference` alone unless asked.
- Commits: the user commits. Give one-liner `git add ... && git commit -m
  "<lean subject>" -m "<descriptive body>"` per semantic change, plus one
  combined command. **No `Co-Authored-By` line.**
- Docs in plain, readable English. Measurements come with how they were taken.
- Verify before claiming: compile checks, accuracy check, and device pass/fail
  criteria.
