# Performance Comparison

## Experiment 1 Results

| Experiment | Configuration | Execution Time (seconds) | Verification |
|---|---|---:|---|
| Sequential | CPU Sequential | 135.727671 | C[0][0] = 4000.00 |
| OpenMP | 8 OpenMP Threads | 123.056779 | C[0][0] = 4000.00 |
| MPI | 4 VMware Ubuntu VMs / 4 MPI Processes | 226.167575 | C[0][0] = 4000.00 |
| CUDA | Not completed | — | — |

## Summary

The Sequential implementation provides the CPU baseline.

The OpenMP implementation uses 8 threads and achieved a lower execution time than the Sequential implementation.

The MPI implementation uses 4 Ubuntu VMware virtual machines with 4 MPI processes.

CUDA is included in the project structure, but no CUDA execution time is reported until it is verified.

## Recorded Results

### Sequential

```text
Matrix multiplication completed!
Execution Time = 135.727671 seconds
C[0][0] = 4000.00

OpenMP matrix multiplication completed!
Execution Time = 123.056779 seconds
C[0][0] = 4000.00

MPI matrix multiplication completed!
Execution Time = 226.167575 seconds
C[0][0] = 4000.00