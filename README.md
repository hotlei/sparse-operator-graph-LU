# sparse-operator-graph-LU

The goal of this repo is to provide an easy to understand solution with reasonable performance

to run the software:

make 

./solve path_to_file.mtx

feature of code:

1.block decompose matrix operations into list / graph of unit operations on 64x64 matrix, then evenly distribute to openMP thread for balanced calculation

2.multi level buffer in memory allocation

3.no pivoting in block decompose, not easy. still good for certain sparse matrix up to 1M x 1M

4.majority of unit operations are multiplication of 64x64 matrix, which I can not get much faster than the 3 loop implementation

5.not memory efficient, put swap on NVME SSD helps
