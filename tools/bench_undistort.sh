#!/usr/bin/env bash
# Undistortion benchmark ON THE JETSON, on a live MJPEG stream: time per frame
# of the nvivafilter library, and the time each pipeline element holds a frame
# with no undistortion, with the library and with the nvundistort element.
# Runs everything twice, at default clocks and with jetson_clocks. About 6 minutes.
#
#   tools/bench_undistort.sh
#
# Needs sudo (jetson_clocks) and both builds (`make`). Settings, as environment
# variables:
#   SRC     source up to the JPEG stream (default: tcpclientsrc host=$HOST port=$PORT)
#   HOST, PORT   address of an MJPEG-over-TCP server (default 192.168.10.1:8881)
#   PARAMS  parameter file (default: calibration/imx708_intrinsics_nvundistort.txt)
#   SECS    seconds per run (default 25)
#
# Every run is a new client. A server that exits when its client disconnects
# (rpicam-vid --listen does) has to be restarted in a loop, e.g. on a Raspberry Pi:
#   while true; do rpicam-vid -t 0 --width 4608 --height 2592 --framerate 14 --codec mjpeg \
#     --autofocus-mode manual --lens-position 0 --flush --listen -o tcp://0.0.0.0:8881; sleep 1; done
#
# Output in /tmp/bench_undistort: summary.txt plus the raw tracer logs.
#   lib:    mean library time per frame (NVUNDISTORT_STATS), first window dropped
#   tracer: median time each element holds a frame (GStreamer latency tracer)

set -u
cd "$(dirname "$0")/.."
LIB=$PWD/src/libnvundistort.so
PLUGIN_DIR=$PWD/src/element
OUT=/tmp/bench_undistort
SECS=${SECS:-25}
SRC=${SRC:-tcpclientsrc host=${HOST:-192.168.10.1} port=${PORT:-8881}}
export NVUNDISTORT_PARAMS=${PARAMS:-$PWD/calibration/imx708_intrinsics_nvundistort.txt}
export GST_PLUGIN_PATH=$PLUGIN_DIR${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}
mkdir -p "$OUT" && : > "$OUT/summary.txt"
for f in "$LIB" "$PLUGIN_DIR/libgstnvundistort.so" "$NVUNDISTORT_PARAMS"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

# One run of the stream for $SECS seconds; $1 = "lib" for the nvivafilter
# library, "element" for the nvundistort element (which takes the decoder's
# output directly, without nvvidconv), or empty for no undistortion.
stream() {
  local nv12='video/x-raw(memory:NVMM),format=NV12'
  local filt=(! nvvidconv ! "$nv12")
  [ "$1" = lib ] && filt+=(! nvivafilter cuda-process=true "customer-lib-name=$LIB" ! "$nv12")
  [ "$1" = element ] && filt=(! nvundistort "params-file=$NVUNDISTORT_PARAMS" ! "$nv12")
  sleep 4  # give a restarting stream server time to come back after the last client
  # shellcheck disable=SC2086  # $SRC is a pipeline fragment, split on purpose
  timeout -s INT "$SECS" gst-launch-1.0 $SRC ! jpegparse \
    ! nvv4l2decoder mjpeg=1 disable-dpb=true enable-max-performance=true \
    "${filt[@]}" ! fakesink sync=false
}

lib_stats() {  # $1 = clock label
  for m in map full; do
    r=$(NVUNDISTORT_STATS=50 NVUNDISTORT_MODE=$m stream lib 2>&1 \
        | grep -o '[0-9]*[.,][0-9]* ms/frame' | tail -n +2 | tr , . \
        | LC_ALL=C awk '{s += $1; n++} END {if (n) printf "%.2f ms/frame (%d windows)", s / n, n}')
    # LC_ALL=C: under a comma-decimal locale awk reads "9.75" as 9.
    echo "lib     $1  $m  ${r:-NO DATA - is the stream up?}" | tee -a "$OUT/summary.txt"
  done
}

tracer() {  # $1 = clock label
  for run in base: off:lib full:lib element:element; do
    name=${run%%:*}
    GST_TRACERS='latency(flags=element)' GST_DEBUG=GST_TRACER:7 \
      GST_DEBUG_FILE="$OUT/lat_$1_$name.log" NVUNDISTORT_MODE=$name stream "${run#*:}" >/dev/null 2>&1
    echo "tracer  $1  $name  done"
  done
}

sudo -v || exit 1
lib_stats default
tracer default
sudo jetson_clocks --store "$OUT/clocks.conf" && trap 'sudo jetson_clocks --restore "$OUT/clocks.conf"' EXIT
sudo jetson_clocks
lib_stats pinned
tracer pinned

python3 - "$OUT" <<'PYEOF' | tee -a "$OUT/summary.txt"
import glob, re, statistics, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/lat_*.log")):
    d = {}
    for m in re.finditer(r"element-latency.*?element=\(string\)(\w+).*?time=\(guint64\)(\d+)", open(f).read()):
        d.setdefault(m[1], []).append(int(m[2]) / 1e6)
    cells = "  ".join(f"{k} {statistics.median(v):.2f}" for k, v in d.items() if not k.startswith("capsfilter"))
    print(f"tracer  {f.split('/')[-1][4:-4]:<16} {cells or 'NO DATA'}   (ms, median)")
PYEOF
