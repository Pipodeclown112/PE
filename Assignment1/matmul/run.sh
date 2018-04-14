#!/bin/bash

make clean; make
prun -np 1 ./matmul_cpu matrices/arc130.mtx matrices/arc130.mtx results/arc_out.mtx
prun -np 1 ./matmul_cpu matrices/ash958.mtx matrices/ash958_trans.mtx results/ash_out.mtx
prun -np 1 ./matmul_cpu matrices/bcspwr06.mtx matrices/bcspwr06.mtx results/bs_out.mtx
prun -np 1 ./matmul_cpu matrices/ch7-7-b1.mtx matrices/ch7-7-b1_trans.mtx results/ch7_out.mtx
prun -np 1 ./matmul_cpu matrices/fs_760_1.mtx matrices/fs_760_1.mtx results/fs_out.mtx
