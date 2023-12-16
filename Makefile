src = $(wildcard *.cpp)
cusrc = $(wildcard *.cu)
obj = $(src:.cpp=.o) $(cusrc:.cu=.o)
export OMP_NUM_THREADS=6

#CCFLAGS = -c -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3 -march=skylake-avx512

#LDFLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

LDFLAGS = -std=c++11 -arch=sm_35 -lm -rdc=true -lcudadevrt -lgomp -D_GLIBCXX_PARALLEL
CCFLAGS = -c -std=c++11 -O3 -arch=sm_35 -rdc=true -Xcompiler -fopenmp -D_GLIBCXX_PARALLEL
#CCFLAGS = -c -std=gnu++14 -O3 -march=skylake-avx512 -fopenmp -march=native -D_GLIBCxx_PARALLEL
#LDFLAGS = -std=c++11 -lm 
#CCFLAGS = -c -std=c++11 -g 
#CUDAFLAGS = --ptxas-options=-v -Xptxas -dlcm=ca --maxrregcount=32
HOST_COMPILER ?= g++ 
CPP = nvcc -ccbin $(HOST_COMPILER)

solve: $(obj)
	$(CPP) -o $@ $^ $(LDFLAGS)

MatrixStdDouble.o: MatrixStdDouble.cpp
	$(CPP) $^ $(CCFLAGS)

BlockPlanner.o: BlockPlanner.cpp
	$(CPP) $^ $(CCFLAGS)

cudaPlan.o: cudaPlan.cpp
	$(CPP) $^ $(CCFLAGS)

BlockReOrder.o: BlockReOrder.cpp
	$(CPP) $^ $(CCFLAGS)

broker.o: broker.cpp
	$(CPP) $^ $(CCFLAGS)

data.o: data.cpp
	$(CPP) $^ $(CCFLAGS)

Matrixcuda.o:Matrixcuda.cu
	$(CPP) $^ $(CCFLAGS) --ptxas-options=-v -Xptxas -dlcm=ca 

main.o: main.cpp
	$(CPP) $^ $(CCFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) solve
