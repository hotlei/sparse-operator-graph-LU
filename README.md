# sparse-operator-graph-LU

The goal of this repo is to provide an easy to understand linear system solver with comparable performance

to run the software:

make 

./solve path_to_file.mtx

branches:

main -- should compile with any c++ compiler, and run on any system

vector -- only tested with g++-9, need AVX-512 instruction set, runs on i7-9800X, 10th generation 10 nm processor and later

cuda -- runs on nvidia card

feature of code:

1.block decompose matrix operations into list / graph of unit operations on 64x64 matrix, then evenly distribute to openMP thread for balanced calculation

2.multi level buffer in memory allocation

3.no pivoting in block decompose. good for certain sparse matrix up to 1M x 1M

4.majority of unit operations are multiplication of 64x64 matrix, my code is not much faster than O2 optimized 3 loop implementation, which is both disappointment and achievement

5.bigger memory footprint, put swap on NVME SSD helps
