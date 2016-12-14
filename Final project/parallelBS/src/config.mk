ifndef CUDA_HOME
    CUDA_HOME := /usr/local/cuda
endif

ARCH := $(shell uname -m)

ifeq ($(ARCH), i386)
    CUDA_LIB_PATH := $(CUDA_HOME)/lib
else
    CUDA_LIB_PATH := $(CUDA_HOME)/lib64
endif
