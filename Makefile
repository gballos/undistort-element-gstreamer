# Builds everything ON THE JETSON: see README.md.
#
#   make            # the nvundistort element and the nvivafilter library
#   make element    # only the element   (src/element/libgstnvundistort.so)
#   make library    # only the library   (src/libnvundistort.so)

all library element clean:
	$(MAKE) -C src $@

.PHONY: all library element clean
