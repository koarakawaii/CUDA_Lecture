CUDIR     := /usr/local/cuda
GSLDIR	  := /home/lab/gsl

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -I$(GSLDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v --compiler-options -fopenmp

BIN     := ising_model_1gpu.exe
FILE	:= ising_model_1gpu.cu
LIB     := -lgsl -lcudadevrt -lgslcblas -lcurand

all: $(BIN)

$(BIN): $(FILE)
	$(NVCC) -o $(BIN) $(NVFLAGS) $(FILE) $(LIB)

clean:
	rm -f $(BIN)
