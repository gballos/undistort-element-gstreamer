# nvundistort: dataflow

How a camera frame travels from the decoder to the undistorted frame, with and
without the `nvundistort` element, and what happens inside the element.

Times are the median each element holds a frame (GStreamer latency tracer) on
the Jetson Orin Nano at 4608×2592 / 14 fps, default clocks, a saved frame
looped. See [ELEMENT_PLAN.md](ELEMENT_PLAN.md) for how they were taken.

## The pipeline, before and after

`nvv4l2decoder mjpeg=1` outputs I420. Converting that to NV12 costs about 22 ms
in whichever element does it. The element does the conversion inside the
remap, so that step disappears.

```mermaid
flowchart TB
  subgraph today["With nvvidconv + nvivafilter: 63.6 ms from decode to undistorted frame"]
    direction LR
    A0["tcpclientsrc<br/>jpegparse"] -->|"JPEG"| A1["nvv4l2decoder<br/>19.6 ms"]
    A1 -->|"I420<br/>full range"| A2["nvvidconv<br/>21.3 ms"]
    A2 -->|"NV12"| A3["nvivafilter<br/>libnvundistort.so<br/>22.7 ms"]
    A3 -->|"NV12<br/>16-235"| A4(["tee"])
  end

  subgraph element["With the nvundistort element: 26.2 ms"]
    direction LR
    B0["tcpclientsrc<br/>jpegparse"] -->|"JPEG"| B1["nvv4l2decoder<br/>19.9 ms"]
    B1 -->|"I420<br/>full range"| B2["nvundistort<br/>6.3 ms"]
    B2 -->|"NV12<br/>full range"| B3(["tee"])
  end

  today ~~~ element
```

After the tee nothing changes: the detection branch (`nvstreammux` 960×544 →
PeopleNet → NVDCF) and the gaze branch (RGBA in system memory → appsink) both
receive the undistorted NV12 frame, so they share one geometry.

## Inside the element, per frame

All frame data stays in NVMM (GPU-visible) memory. The element never copies
the input: the kernels read the decoder's buffer through textures and write
into a different buffer from the element's own pool.

```mermaid
flowchart TD
  DEC["Decoder pool<br/>4 NVMM buffers, I420"] -->|"input buffer"| LOOK{"CUDA mapping kept<br/>for this buffer?"}
  LOOK -->|"yes"| TEX["3 textures<br/>Y, U, V planes"]
  LOOK -->|"no: first time, about 2 ms"| MAP["Map through EGL<br/>register with CUDA<br/>create textures"]
  MAP --> TEX

  PAR["params-file<br/>K, new K, k1-k6, p1, p2"] --> KY
  PAR --> KUV

  TEX -->|"Y"| KY["remap_y<br/>full resolution"]
  TEX -->|"U and V"| KUV["remap_uv_planar<br/>half resolution"]

  POOL["Output pool<br/>4 to 8 NVMM buffers, NV12"] -->|"output buffer<br/>mapped once, kept"| OUT
  KY -->|"Y plane"| OUT["Undistorted NV12 frame"]
  KUV -->|"interleaved UV plane"| OUT
  OUT --> DOWN(["downstream: caps filter, tee"])

  DEC -.->|"only if undistortion is disabled:<br/>plain I420 to NV12 conversion"| OUT
```

- **Kept mappings.** Mapping a buffer into CUDA and unmapping it again costs
  about 3 ms, so each buffer is mapped once and the mapping is kept, keyed by
  the buffer's dmabuf fd. The decoder cycles through four buffers. Mappings
  are dropped on a caps change and when the element stops.
- **The dashed path** is the fallback. If undistortion cannot run (no usable
  `params-file`, another frame size, a CUDA error), the element warns once and
  only converts, so frames keep flowing, distorted.

## Inside a kernel, per output pixel

Both kernels do the same thing, `remap_y` on luma and `remap_uv_planar` on
the half-resolution chroma. There is no lookup table: the source position is
computed for every pixel, with the same model as OpenCV's
`initUndistortRectifyMap`.

```mermaid
flowchart LR
  P["Output pixel<br/>(u, v)"] --> UN["Unproject<br/>with new K"]
  UN --> LENS["Apply the lens model<br/>rational radial + tangential"]
  LENS --> PR["Project<br/>with K"]
  PR --> SRC["Source position<br/>(xs, ys)"]
  SRC --> IN{"Inside the<br/>source frame?"}
  IN -->|"yes"| FETCH["Texture fetch<br/>bilinear blend"]
  IN -->|"no"| BORDER["Black<br/>Y = 0, U = V = 128"]
  FETCH --> W["Write the pixel"]
  BORDER --> W
```
