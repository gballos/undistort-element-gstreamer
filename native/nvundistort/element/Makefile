# Builds libgstnvundistort.so, the nvundistort GStreamer element, ON THE JETSON
# (Orin = sm_87). Needs DeepStream and the GStreamer development packages.
#
#   make -C native/nvundistort/element
#   GST_PLUGIN_PATH=$PWD/native/nvundistort/element gst-inspect-1.0 nvundistort
#
# Two compilers: nvcc for the CUDA side, g++ for the GStreamer side (nvcc and
# the glib headers do not mix).

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_87
DS        ?= /opt/nvidia/deepstream/deepstream
MMAPI     ?= /usr/src/jetson_multimedia_api/include
PKGS      := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CXXFLAGS  := -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -I$(DS)/sources/includes \
             $(shell pkg-config --cflags $(PKGS))
LIBS      := -L$(CUDA_HOME)/lib64 -lcudart -L/usr/lib/aarch64-linux-gnu/tegra -lnvbufsurface \
             -lnvbufsurftransform -lEGL -L$(DS)/lib -lnvdsbufferpool -Wl,-rpath,$(DS)/lib \
             $(shell pkg-config --libs $(PKGS))

libgstnvundistort.so: gstnvundistort.o undistort_engine.o
	$(CXX) -shared -Wl,-no-undefined $^ $(LIBS) -o $@

gstnvundistort.o: gstnvundistort.cpp undistort_engine.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

undistort_engine.o: undistort_engine.cu undistort_engine.h ../undistort_kernels.cuh ../undistort_params.h
	$(NVCC) -O3 -arch=$(ARCH) -Xcompiler -fPIC -I$(MMAPI) -c $< -o $@

clean:
	rm -f libgstnvundistort.so gstnvundistort.o undistort_engine.o

.PHONY: clean
