# Plan: replace nvivafilter with our own undistortion element

Hand-off document for a new work session. It is self-contained: the current
state, the measurements behind the decision, the design, the step-by-step plan
with pass/fail gates, and the working conventions.

**Status:** optional optimisation. Undistortion works and is correct today.
This plan removes about 30 ms of per-frame latency and a slight image
softening. It is not a bug fix.

---

## 1. Why

Measured on the Jetson Orin Nano, IMX708 at 4608×2592 / 14 fps, camera stream
only (no inference running), clocks pinned with `jetson_clocks`. The GStreamer
latency tracer gives the median time each element holds a frame:

| Stage | Today, undistortion on | Today, off | With our element (estimate) |
|---|---|---|---|
| `nvv4l2decoder` | 19.9 ms | 19.5 ms | ~20 ms |
| `nvvidconv` (NV12 copy, no flip) | 21.5 ms | 21.6 ms | **gone**, if our element accepts decoder output (see §4) |
| `nvivafilter` overhead (its internal copy) | 11.1 ms | — | **gone** |
| Our CUDA library (map + scratch copy + remap) | 4.2 ms | — | ~2–3 ms (no scratch copy; mapping cached) |
| **Decode → undistorted frame** | **~57 ms** | ~42 ms | **~23–25 ms** |

At default (unpinned) clocks, everything we control is 2–2.5× slower: the
library takes ~10 ms, and today's total is ~66 ms.

- **The two stages around our code cost ~32 ms** (`nvvidconv` 21.5 ms plus
  `nvivafilter`'s own copy 11 ms), and clock speed doesn't change them. Our
  library itself is ~4 ms and has little left to optimise.
- `nvivafilter`'s copy also **softens the image**: 4.8 brightness levels mean
  absolute change on luma, with no overall shift, even when our library does
  nothing.
- **In context:** the documented glass-to-glass budget is ~214 ms, of which the
  Pi's capture + JPEG encode is ~164 ms (wiki page *Gaze-Pipeline-Performance*,
  measured on the older IMX219). That budget omits the `nvvidconv` and
  undistortion stages, so the real total today is higher. The element would
  cut roughly 30 ms, about 12–15% of the end-to-end latency, and give sharper
  frames.

## 2. Current state

**Repo:** `~/DOOH/src/edge-inference`, branch `feat/undistortion_integration`,
not pushed. Relevant commits, oldest first: `c393a37` (library), `d5b015e`
(params writer + check tool), `c7d33fc` (pipeline wiring), `e0171eb` (UV pitch
+ locale fixes), `f6a106a` (check reference), `c2d266b` (flip dropped, filter
after `nvvidconv`), `947cd6e` (params file committed), `6e54d82` (kernels in a
shared header), `a03e88d` (texture sampling). `tools/bench_undistort.sh` is
still **uncommitted**.

**Pipeline** (`pipeline/builder.py`):
```
tcpclientsrc → jpegparse → nvv4l2decoder mjpeg=1 → nvvidconv → caps NV12 NVMM
  → [nvivafilter cuda-process=true customer-lib-name=libnvundistort.so → caps NV12 NVMM]  (only if calibration_npz set)
  → tee → detection branch (nvstreammux 960x544 → PeopleNet → NVDCF) / gaze branch (RGBA system memory → appsink)
```
Undistortion must stay **before the tee**: the assembler scales detection
boxes onto the gaze frame, so both branches need the same geometry.

**Files:**

| File | Role |
|---|---|
| `native/nvundistort/undistort_kernels.cuh` | `Params`, `src_coords` (exact OpenCV rational + tangential model), `make_plane_texture`, kernels `remap_y` / `remap_uv` (texture in, pitched NV12 out). **Reused unchanged by the element.** |
| `native/nvundistort/nvundistort.cu` | `nvivafilter` wrapper: loads params, maps the EGLImage, checks the surface, copies to scratch, launches, syncs. Env: `NVUNDISTORT_PARAMS`, `NVUNDISTORT_MODE=off\|map\|y\|full`, `NVUNDISTORT_STATS=N`. |
| `native/nvundistort/Makefile`, `README.md` | Build on the Jetson; needs NVIDIA's `customer_functions.h` in `include/`. |
| `utils/undistort_params.py` | `.npz` → `key value` text params file (`write_params`, `load_intrinsics`). |
| `imx708_intrinsics.npz`, `imx708_intrinsics_nvundistort.txt` | Calibration (alpha 0) and its params file, both committed. |
| `config.py` / `config.yaml` | `camera.<profile>.calibration_npz` enables undistortion; `undistort.alpha`. |
| `tools/check_undistort.py` | On-Jetson accuracy check against `cv2.remap`. The reference input is currently `nvivafilter` in `mode=off`, because nvivafilter alters frames. |
| `tools/bench_undistort.sh` | Library A/B + per-element tracer at default and pinned clocks, ~7 min. |

**Hard-won facts (do not rediscover these):**
- The Jetson reports NV12's UV plane pitch in **U,V pairs** (2304 for
  4608-byte rows). `uv_pitch_bytes()` converts it.
- `gst-launch` applies the system locale (comma decimal). `sscanf` and `awk`
  must parse in the C locale (`uselocale`, `LC_ALL=C`).
- `nvivafilter` **does not negotiate with `nvv4l2decoder` directly**
  (not-negotiated). That's why `nvvidconv` sits in front of it.
- `nvdewarper` was tried and abandoned: it's for 360°/fisheye cameras, gave
  wrong geometry, and was slow.
- Texture fetches need `+0.5` on coordinates (texel centres), and plane
  pointers/pitches must be texture-aligned (`cudaMallocPitch`).
- Accuracy today: mean 0.03 levels vs OpenCV, p99 1 level. The only larger
  differences are in the outermost 1-px ring.
- The calibration is valid only for **4608×2592, no rotation, focus locked**:
  the Pi runs `rpicam-vid -t 0 --width 4608 --height 2592 --framerate 14
  --codec mjpeg --autofocus-mode manual --lens-position 0 --flush --listen -o
  tcp://192.168.10.1:8881`.
  `--listen` serves one client and exits, so for repeated tests wrap it in
  `while true; do ...; sleep 1; done`.

## 3. What the element must do

One GStreamer element, `dec → [element] → tee`, that:
1. accepts `nvv4l2decoder`'s NVMM NV12 output **directly**;
2. reads the decoder's frame as the texture source and writes the undistorted
   frame into a **separate** output NVMM NV12 buffer from its own pool. This
   means no `nvvidconv`, no `nvivafilter` copy and no scratch copy;
3. keeps the downstream contract unchanged: `video/x-raw(memory:NVMM),format=NV12`,
   same size, pitch-linear. `nvvidconv` produces exactly this today;
4. caches the CUDA registration and texture objects per buffer. This is safe
   here because we own the output pool and can confirm the decoder's pool
   recycles;
5. keeps the existing failure behaviour: any CUDA error disables the
   correction, logs once, and passes frames through;
6. is driven by the same params file and `config.yaml` selection.

**Proposed basis:** DeepStream 7.1's `nvdsvideotemplate`. It's a ready-made
GStreamer element that loads a C++ "custom library" (`customlib-name=...`,
settings via `customlib-props="key:value"`). The library implements NVIDIA's
`IDSCustomLibrary` interface (init params, properties, events, `SubmitInput`
per buffer) and produces its own output buffers. **The exact interface and
buffer-pool API must be read from the installed sources** (§5, step 0). They
haven't been read yet, and the names above are from memory.

## 4. The key technical unknown: the decoder's memory layout

On Jetson, `nvv4l2decoder` often outputs **block-linear** NVMM surfaces (the
hardware's tiled layout). Block-linear would explain two things we observed:
- **`nvivafilter` refuses the decoder's output.** It expects pitch-linear.
- **`nvvidconv` costs 21.5 ms for a "same-size copy".** It is probably
  converting block-linear to pitch-linear.

**Why this suits our design:** when a block-linear surface is mapped through
EGL, CUDA gives it as a **CUDA array** (`cudaEglFrameTypeArray`), not a pitched
pointer. Our kernels already read through **textures**, and a texture can be
built on a CUDA array (`cudaResourceTypeArray`) just as easily as on pitched
memory. So the kernels stay unchanged; only `make_plane_texture` needs an
array variant. The output buffer is allocated pitch-linear, so downstream is
unaffected.

**To confirm first:** feed the decoder's surface to the existing
`surface_ok()` log (`frameType=0` means array/block-linear, `1` means pitch).
The spike (step 1) prints this.

## 5. Plan, with gates

Each step is one commit and one device command with a clear pass/fail. The
user runs all device steps; there is no SSH access from the dev machine.

**Step 0 — Probe (15 min).** On the Jetson:
```bash
gst-inspect-1.0 nvdsvideotemplate | head -60
ls /opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvdsvideotemplate/
tar czf /tmp/ds_src.tgz -C /opt/nvidia/deepstream/deepstream-7.1/sources includes gst-plugins/gst-nvdsvideotemplate
```
Copy `ds_src.tgz` to the dev machine for local compile checks. Read the
interface header and the sample custom library before designing anything.
*Gate:* the element exists and its sink caps accept NVMM NV12.

**Step 1 — Spike with NVIDIA's sample library (½ day, throwaway).** Build the
sample custom lib unchanged and answer:
1. Does `nvv4l2decoder ! nvdsvideotemplate customlib-name=<sample>` negotiate
   **without** `nvvidconv`?
2. What does the element cost doing nothing? Use the tracer, like
   `nvivafilter` `off` (11.1 ms today). It must be well below that.
3. Is its pass-through byte-identical to the decode (`cmp`)? `nvivafilter`'s
   isn't.
4. What memory layout does the decoder hand over (§4)?

*Gate:* if (1) fails, `nvvidconv` stays and the saving shrinks to ~11 ms +
softening; decide whether that is still worth it. If (2) is not clearly lower,
stop.

**Step 2 — The custom library (2–3 days).** New
`native/nvundistort/nvdsundistort.cpp` (or `.cu`) built alongside the current
library, including `undistort_kernels.cuh`. It includes: per-buffer cached
registration + textures, an output pool, params via `customlib-props` (or
reusing `NVUNDISTORT_PARAMS`), `NVUNDISTORT_STATS`, and the same
disable-on-error behaviour. Compile locally for `sm_87` before every device
round-trip (§6).

**Step 3 — Accuracy check.** Add a way for `tools/check_undistort.py` to use
the element (an option alongside `--lib`). The reference becomes the **plain
decode**, since the element must not alter frames. That's a stricter test
than today's. *Pass:* mean < 1 level on Y/U/V, similar to today's 0.03.

**Step 4 — Builder.** In `pipeline/builder.py`, `_insert_undistort` links
`dec → element → caps` instead of `vidconv_caps → nvivafilter → caps`.
Keep `nvivafilter` as a fallback until step 5 passes, then remove it in one
commit.

**Step 5 — Evaluate.** Run `tools/bench_undistort.sh` (extended for the
element), then the app-level tracer runs and the 30-min soak (§7). *Pass:*
decode → undistorted frame ≈ 25 ms pinned, no `UNDISTORTION DISABLED`, flat
memory, no CUDA errors.

**Step 6 — Docs.** Update the wiki pages (§8) and the library README.

**Risks:** buffer-pool bugs (leaks, stalls) are the classic failure; the
DeepStream API may change on upgrade; the builder notes crashes when both tee
branches shared a transform engine (`DET_COMPUTE_HW` comment), so soak-test
anything that adds GPU work before the tee.

## 6. Local build setup (dev machine, no GPU)

Compile checks run locally with NVIDIA's CUDA 12.6 redistributables (no
install needed). From
`https://developer.download.nvidia.com/compute/cuda/redist/` (index
`redistrib_12.6.3.json`), unpack `cuda_nvcc-linux-x86_64-12.6.85-archive` and
`cuda_cudart-linux-x86_64-12.6.77-archive` into one directory (`bin/`,
`include/`, `lib/`), then:
```bash
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC,-Wall,-Wextra -I<stub-or-ds-includes> -I. <file> -L<root>/lib -o /tmp/x.so
```
- For the `nvivafilter` library, a stand-in `customer_functions.h` is enough
  for compiling. It must declare `CustomerFunction` with `fPreProcess`,
  `fGPUProcess` and `fPostProcess`, and `ColorFormat`. Include it inside
  `extern "C" {}`.
- For the element, use the DeepStream headers from step 0.
- **Proving a refactor changes nothing:** compare `-ptx` output before and
  after, ignoring mangled symbol names. That's how `6e54d82` was verified.
- **Predicting accuracy without a GPU:** numpy simulations of the kernels
  have matched device results. Also check that `init`/`deinit` stay exported
  unmangled (`nm -D`).

## 7. Still open (independent of the element)

- **App-level measurements** (commands prepared, not run yet): the tracer on
  `python -m apps.run` with undistortion on and off
  (`GST_TRACERS='latency(flags=pipeline+element)'`, `PIPELINE_LAT_CSV`), and a
  30-min soak with `tegrastats`. These show whether undistortion competes with
  inference for the GPU. **Note:** the `utils/latency` suite cannot see
  undistortion. Its round-trip runs bypass DeepStream, and its stage timing
  starts after the appsink.
- **Clock policy for production:** see §9. Decide whether the deployed Jetson
  runs `jetson_clocks` at boot, and record the `nvpmodel` mode.
- **`utils/csi_camera.py` (gaze-calibration app) does not undistort,** so
  gaze-to-screen calibrations recorded with it don't match production
  geometry.
- **Without undistortion, `nvvidconv` may be removable** (saves ~21.5 ms), but
  it may also be what frees the decoder's buffers quickly. Needs its own test.
- **IMX219 orientation:** its old docs rotated on the Pi *and* flipped on the
  Jetson; now it does neither. Check before switching back to it.
- **Calibration focus:** confirm the calibration frames were shot with the
  focus locked (`--lens-position 0`).

## 8. Documentation locations

- Wiki (git repo, Azure DevOps format, `%2D` = `-` in file names, `.order`
  files list page order): `~/DOOH/docs/ELIAS---AdaptiveSight.wiki/`.
  - `Infrastructure/Camera-Stream-%2D-Infrastructure-and-Consumption-Pipeline.md` (AS-BUILT)
  - `.../Camera-Stream-.../pipeline%2Ddiagram.md` (mermaid diagram + GPU/CPU table)
  - `.../Camera-Stream-.../Lens%2DUndistortion.md` (process, performance, inefficiencies, limitations)
- Library docs: `native/nvundistort/README.md`.

## 9. Pinned vs default clocks, in short

`jetson_clocks` locks the CPU, GPU and memory controller at their maximum
frequency within the current power mode (`nvpmodel`), instead of letting them
scale with load. Our library does a short burst of work every 71 ms, and at
default clocks the GPU stays mostly at low frequency. That's why the same code
takes ~10 ms at default clocks and ~4 ms pinned. The video hardware (decoder,
VIC in `nvvidconv`, `nvivafilter`'s copy) barely changes. Under full inference
load the GPU clocks up by itself, so production likely sits between the two.
The app-level runs (§7) will show where. Pinning at boot costs idle power and
heat. It isn't persistent unless started by a systemd unit.

## 10. Working conventions

- Make the smallest change that does the job. Match the surrounding code's
  style and comment density.
- The user runs every Jetson/Pi command; give exact commands and what output
  to expect.
- Commits: the user commits. Give one-liner `git add ... && git commit -m
  "<lean subject>" -m "<descriptive body>"` per semantic change, plus one
  combined command. **No `Co-Authored-By` line.**
- Docs in plain, readable English. Measurements come with how they were taken.
- Verify before claiming: compile checks, PTX diffs, simulations, and device
  pass/fail criteria.
