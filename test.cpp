#include <cstdio>
#include <mpi.h>
#include <omp.h>


int main(){

    int nprocs, rank;
    int tid, nthreads;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel
    printf("Proc %d/%d, thread %d/%d\n", rank, nprocs, omp_get_thread_num(), omp_get_max_threads());

    MPI_Finalize();
    return 0;


}