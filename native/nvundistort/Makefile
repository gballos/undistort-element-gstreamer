# Builds libnvundistort.so ON THE JETSON (Orin = sm_87). See README.md.
#
#   make -C native/nvundistort
#
# Needs NVIDIA's customer_functions.h (the nvivafilter plug-in interface) in
# include/ -- it is not vendored here; README.md says where to get it.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_87
HEADER    := include/customer_functions.h

libnvundistort.so: nvundistort.cu undistort_kernels.cuh undistort_params.h $(HEADER)
	$(NVCC) -O3 -arch=$(ARCH) -shared -Xcompiler -fPIC -Iinclude $< -o $@

$(HEADER):
	@echo "missing $(HEADER): copy NVIDIA's customer_functions.h there (see README.md)" >&2
	@exit 1

clean:
	rm -f libnvundistort.so

.PHONY: clean
