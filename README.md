# sparse-operator-graph-LU

The goal of this repo is to provide an easy to understand linear system solver with reasonable performance

to run the software:

make 

./solve path_to_file.mtx

branches:

main -- should compile with any c++ compiler, and run on any system

vector -- only tested with g++-9, compiled with avx 512 instruction, runs on X series processor or 10th generation 10 nm processor and later

feature of code:

1.block decompose matrix operations into list / graph of unit operations on 64x64 matrix, then evenly distribute to openMP thread for balanced calculation

2.two level buffer in memory allocation, no confirmed memory leak

3.no pivoting in block decompose, will try later. still good for certain sparse matrix up to 1M x 1M

4.majority of unit operations (/ hot spot) are multiplication of 64x64 matrix, which I can not get much faster than the 3 loop implementation

5.not memory efficient, use 5 times more than the better solvers for certain example, can and will be cut in half. put swap on NVME SSD helps
