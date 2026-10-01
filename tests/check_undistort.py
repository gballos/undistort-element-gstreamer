"""Accuracy check for native/nvundistort ON THE JETSON: runs one JPEG through
the undistortion and compares the result with cv2.remap.

    cd tests && python3 check_undistort.py frame.jpg --npz ../imx708_intrinsics.npz            # nvivafilter library
    cd tests && python3 check_undistort.py frame.jpg --npz ../imx708_intrinsics.npz --element  # nvundistort element

For the library, the reference input is the same chain with the library in
mode=off, so any difference is the library's alone. For the element it is the
plain decode: the element must not alter a frame beyond undistorting it.
PASS = mean abs difference below 1 level on every plane.
Use a frame shot at the calibrated resolution and orientation (4608x2592, no
rotation, focus locked -- see the top-level README).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from undistort_params import load_intrinsics, write_params

DEFAULT_LIB = Path(__file__).resolve().parent.parent / "native/nvundistort/libnvundistort.so"
DEFAULT_PLUGIN_DIR = DEFAULT_LIB.parent / "element"


def run_nv12(jpeg, out, lib=None, env=None):
    """Decode `jpeg` through NVMM NV12, optionally through nvivafilter, and
    write the raw NV12 frame to `out`."""
    filt = (f"nvivafilter cuda-process=true customer-lib-name={lib} ! "
            "video/x-raw(memory:NVMM),format=NV12 ! ") if lib else ""
    pipeline = (f"filesrc location={jpeg} ! jpegdec ! nvvidconv ! "
                "video/x-raw(memory:NVMM),format=NV12 ! " + filt +
                f"nvvidconv ! video/x-raw,format=NV12 ! filesink location={out}")
    subprocess.run(["gst-launch-1.0", "-q", *pipeline.split()], check=True, env=env)


def run_element(jpeg, out, plugin_dir, params=None):
    """Decode `jpeg` with nvv4l2decoder, as the pipeline does, optionally through
    the nvundistort element, and write the raw NV12 frame to `out`."""
    elem = (f"nvundistort params-file={params} ! "
            "video/x-raw(memory:NVMM),format=NV12 ! ") if params else ""
    pipeline = (f"filesrc location={jpeg} ! jpegparse ! nvv4l2decoder mjpeg=1 ! " + elem +
                f"nvvidconv ! video/x-raw,format=NV12 ! filesink location={out}")
    env = dict(os.environ, GST_PLUGIN_PATH=str(plugin_dir))
    subprocess.run(["gst-launch-1.0", "-q", *pipeline.split()], check=True, env=env)


def read_nv12(path, w, h):
    raw = np.fromfile(path, np.uint8)
    if raw.size != w * h * 3 // 2:
        sys.exit(f"{path}: {raw.size} bytes, expected {w * h * 3 // 2} for {w}x{h} NV12 "
                 "(row padding? use a width that is a multiple of 64)")
    y = raw[:w * h].reshape(h, w)
    uv = raw[w * h:].reshape(h // 2, w // 2, 2)
    return y, uv[..., 0], uv[..., 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jpeg")
    parser.add_argument("--npz", required=True, help="calibration .npz")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--lib", default=str(DEFAULT_LIB))
    parser.add_argument("--element", action="store_true",
                        help="check the nvundistort element instead of the nvivafilter library")
    parser.add_argument("--plugin-dir", default=str(DEFAULT_PLUGIN_DIR),
                        help="directory holding libgstnvundistort.so (with --element)")
    parser.add_argument("--out-dir", default=".", help="where the raw frames and preview go")
    args = parser.parse_args()

    K, d8, new_K, (w, h) = load_intrinsics(args.npz, args.alpha)
    out_dir = Path(args.out_dir)
    src_raw, dst_raw = out_dir / "undistort_in.nv12", out_dir / "undistort_out.nv12"
    params = write_params(args.npz, args.alpha, out_dir / "undistort_params.txt")
    if args.element:
        plugin_dir = Path(args.plugin_dir).resolve()
        run_element(args.jpeg, src_raw, plugin_dir)
        run_element(args.jpeg, dst_raw, plugin_dir, params=Path(params).resolve())
    else:
        env = dict(os.environ, NVUNDISTORT_PARAMS=params)
        # nvivafilter alters the frame even when the library does nothing, so the
        # reference input is its passthrough output (mode=off), not the plain decode.
        lib = Path(args.lib).resolve()
        run_nv12(args.jpeg, src_raw, lib=lib, env=dict(env, NVUNDISTORT_MODE="off"))
        run_nv12(args.jpeg, dst_raw, lib=lib, env=dict(env, NVUNDISTORT_MODE="full"))

    y, u, v = read_nv12(src_raw, w, h)
    oy, ou, ov = read_nv12(dst_raw, w, h)

    mx, my = cv2.initUndistortRectifyMap(K, d8, None, new_K, (w, h), cv2.CV_32FC1)
    # Chroma sample j sits on luma coordinate 2j + 0.5; a 2x INTER_LINEAR resize
    # samples the luma map exactly there. Then luma L -> chroma (L - 0.5) / 2.
    cmx = (cv2.resize(mx, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) - 0.5) / 2
    cmy = (cv2.resize(my, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) - 0.5) / 2
    ref = {
        "Y": cv2.remap(y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0),
        "U": cv2.remap(u, cmx, cmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128),
        "V": cv2.remap(v, cmx, cmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128),
    }
    ok = True
    for name, got in (("Y", oy), ("U", ou), ("V", ov)):
        diff = np.abs(got.astype(np.int16) - ref[name].astype(np.int16))
        ok &= diff.mean() < 1.0
        print(f"{name}: mean {diff.mean():.3f}  p99 {np.percentile(diff, 99):.0f}  "
              f"max {diff.max()}  >2: {(diff > 2).mean() * 100:.3f}%")
    # Differences confined to the outermost pixel ring are expected: cv2 blends
    # the border colour in there, the kernel does not. With --element, a few
    # chroma pixels on sharp colour edges also differ: the nvvidconv that
    # downloads the reference frame filters chroma there, the element does not.

    nv12 = np.vstack([oy, np.stack([ou, ov], axis=-1).reshape(h // 2, w)])
    preview = out_dir / "undistort_preview.jpg"
    cv2.imwrite(str(preview), cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
    print(f"preview: {preview}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
