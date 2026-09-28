#!/usr/bin/env bash
# Undistortion benchmark ON THE JETSON: library A/B and per-element latency,
# at default and at pinned clocks. About 7 minutes.
#
#   tools/bench_undistort.sh [old_lib.so]    (default old lib: libnvundistort_manual.so)
#
# rpicam-vid --listen exits when its client disconnects, and every run here is a
# new client, so run the Pi in a restart loop first:
#   while true; do rpicam-vid -t 0 --width 4608 --height 2592 --framerate 14 --codec mjpeg \
#     --autofocus-mode manual --lens-position 0 --flush --listen -o tcp://192.168.10.1:8881; sleep 1; done
#
# Output in /tmp/bench_undistort: summary.txt plus the raw tracer logs.
#   A/B:    mean library time per frame (NVUNDISTORT_STATS), first window dropped
#   tracer: median time each element holds a frame (GStreamer latency tracer)

set -u
cd "$(dirname "$0")/.."
LIBDIR=$PWD/native/nvundistort
NEW=$LIBDIR/libnvundistort.so
OLD=${1:-$LIBDIR/libnvundistort_manual.so}
OUT=/tmp/bench_undistort
SECS=${SECS:-25}
export NVUNDISTORT_PARAMS=$PWD/imx708_intrinsics_nvundistort.txt
mkdir -p "$OUT" && : > "$OUT/summary.txt"
for f in "$NEW" "$OLD" "$NVUNDISTORT_PARAMS"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

# One run of the stream for $SECS seconds; $1 = filter library, or empty for none.
stream() {
  local filt=()
  [ -n "$1" ] && filt=(! nvivafilter cuda-process=true "customer-lib-name=$1"
                       ! 'video/x-raw(memory:NVMM),format=NV12')
  sleep 4  # give the Pi's rpicam-vid loop time to restart after the last client
  timeout -s INT "$SECS" gst-launch-1.0 tcpclientsrc host=192.168.10.1 port=8881 ! jpegparse \
    ! nvv4l2decoder mjpeg=1 disable-dpb=true enable-max-performance=true \
    ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' "${filt[@]}" ! fakesink sync=false
}

ab() {  # $1 = clock label
  for lib in "$OLD" "$NEW"; do
    for m in map full; do
      r=$(NVUNDISTORT_STATS=50 NVUNDISTORT_MODE=$m stream "$lib" 2>&1 \
          | grep -o '[0-9]*[.,][0-9]* ms/frame' | tail -n +2 | tr , . \
          | LC_ALL=C awk '{s += $1; n++} END {if (n) printf "%.2f ms/frame (%d windows)", s / n, n}')
      # LC_ALL=C: under a comma-decimal locale awk reads "9.75" as 9.
      echo "A/B     $1  $(basename "$lib")  $m  ${r:-NO DATA - is the Pi stream up?}" | tee -a "$OUT/summary.txt"
    done
  done
}

tracer() {  # $1 = clock label
  for run in base: "off:$NEW" "full:$NEW"; do
    name=${run%%:*}
    GST_TRACERS='latency(flags=element)' GST_DEBUG=GST_TRACER:7 \
      GST_DEBUG_FILE="$OUT/lat_$1_$name.log" NVUNDISTORT_MODE=$name stream "${run#*:}" >/dev/null 2>&1
    echo "tracer  $1  $name  done"
  done
}

sudo -v || exit 1
ab default
tracer default
sudo jetson_clocks --store "$OUT/clocks.conf" && trap 'sudo jetson_clocks --restore "$OUT/clocks.conf"' EXIT
sudo jetson_clocks
ab pinned
tracer pinned

python3 - "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import glob, re, statistics, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/lat_*.log")):
    d = {}
    for m in re.finditer(r"element-latency.*?element=\(string\)(\w+).*?time=\(guint64\)(\d+)", open(f).read()):
        d.setdefault(m[1], []).append(int(m[2]) / 1e6)
    cells = "  ".join(f"{k} {statistics.median(v):.2f}" for k, v in d.items() if not k.startswith("capsfilter"))
    print(f"tracer  {f.split('/')[-1][4:-4]:<14} {cells or 'NO DATA'}   (ms, median)")
EOF
