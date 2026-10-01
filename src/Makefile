# Builds the undistortion ON THE JETSON (Orin = sm_87): the nvivafilter library
# libnvundistort.so and the nvundistort element element/libgstnvundistort.so.
# See the top-level README.md.
#
#   make -C src            # both
#   make -C src library    # only the nvivafilter library
#   make -C src element    # only the element
#
# The library builds against NVIDIA's customer_functions.h (the nvivafilter
# plug-in interface) in include/. If your Jetson Linux release ships a
# different one, replace it; the top-level README.md says where to get it.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_87
HEADER    := include/customer_functions.h

all: library element

library: libnvundistort.so

element:
	$(MAKE) -C element

libnvundistort.so: nvundistort.cu undistort_kernels.cuh undistort_params.h $(HEADER)
	$(NVCC) -O3 -arch=$(ARCH) -shared -Xcompiler -fPIC -Iinclude $< -o $@

clean:
	rm -f libnvundistort.so
	$(MAKE) -C element clean

.PHONY: all library element clean
