src = $(wildcard *.cpp)
obj = $(src:.cpp=.o) 
export OMP_NUM_THREADS=6

CCFLAGS = -c -std=c++11 -O3 -Ofast -fopenmp -fpermissive -march=native -pthread -D_GLIBCXX_PARALLEL
#CCFLAGS = -c -std=c++11 -O3 -Ofast -fopenmp -fpermissive -pthread -D_GLIBCXX_PARALLEL
#CCFLAGS = -c -std=c++11 -g -fopenmp -fpermissive -pthread -D_GLIBCXX_PARALLEL
LDFLAGS = -lm -fopenmp
#CPP = g++-9
CPP = c++

solve: $(obj)
	$(CPP) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CPP) $^ $(CCFLAGS)

clean:
	rm -f $(obj) solve
