include config.mk

PKG_LIBS = -lgomp
PKG_CFLAGS = -fopenmp

#R_INC_PATH ?= -I/usr/include/R -I/usr/include/R/R_ext/
R_INC_PATH ?= -I/usr/include/R -I/home/mrnbdusk/R/Rcpp/include

NVCC = /usr/local/cuda-5.5/bin/nvcc
NVCC_FLAGS = -Xcompiler "-fpic"

# CUDA Libraries -d
LIBS = -lcudart

# R gcc build script
R_CC = R CMD SHLIB

OUT_OBJ = bs_omp.o bs_cuda.o
OUT_SO = bs_parallel.so

all: $(OUT_OBJ)
	$(R_CC) $(OUT_OBJ) -o $(OUT_SO) $(LIBS) -L$(CUDA_LIB_PATH)
	rm -f *.o
bs_omp.o: bs_omp.cpp
	g++ $(R_INC_PATH) $(PKG_LIBS) $(PKG_CFLAGS) "-fpic" -c bs_omp.cpp
bs_cuda.o: bs_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $(R_INC_PATH) -c bs_cuda.cu
