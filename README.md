# Parallel Computing Experiment

## Experiments completed

This repository contains evidence for the parallel computing work completed:

1. **Sequential Matrix Multiplication**
2. **OpenMP Matrix Multiplication**
3. **MPI Matrix Multiplication using VMware Ubuntu virtual machines**

**CUDA/GPU was not completed and is intentionally not included.**

## Results

| Experiment | Configuration | Execution Time | Verification |
|---|---|---:|---:|
| Sequential | CPU sequential | 135.727671 s | C[0][0] = 4000.00 |
| OpenMP | 8 OpenMP threads | 123.056779 s | C[0][0] = 4000.00 |
| MPI | 4 VMware Ubuntu VMs / 4 MPI processes | 226.167575 s | C[0][0] = 4000.00 |

> The MPI screenshots in this repository show the VMware Workstation setup with `master`, `worker1`, `worker2`, and `worker3`. The successful run used 4 MPI processes, with each rank computing 1000 rows.

## OpenMP

Compilation:

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

Threads:

```bash
export OMP_NUM_THREADS=8
```

Execution:

```bash
./matrix_openmp
```

Final recorded result:

```text
OpenMP matrix multiplication completed!
Execution Time = 123.056779 seconds
C[0][0] = 4000.00
```

## MPI

The MPI work included Open MPI and `mpicc`, compilation of the matrix multiplication program, and execution with multiple processes.

Example commands:

```bash
mpicc -O2 -o matrix_mpi matrix_mpi.c
mpirun -np 4 ./matrix_mpi
```

For the college submission, use the VMware screenshots for the final MPI evidence if the lab specifically requires VMware Workstation Pro.

## Sequential

Final recorded result:

```text
Matrix multiplication completed!
Execution Time = 135.727671 seconds
C[0][0] = 4000.00
```

## Repository structure

```text
CC-Parallel-Computing-Experiment/
├── README.md
└── screenshots/
    ├── sequential/
    ├── openmp/
    └── mpi/
```

## CUDA

CUDA/GPU was not completed, so no CUDA result is claimed in this repository.
