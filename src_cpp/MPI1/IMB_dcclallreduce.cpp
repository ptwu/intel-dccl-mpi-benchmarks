/**************************************************************************
 File: IMB_dcclallreduce.c 

 Implemented functions: 

 IMB_dcclallreduce;


 ***************************************************************************/





#include "IMB_declare.h"
#include "IMB_benchmark.h"

#include "IMB_prototypes.h"
#include "dccl_allreduce_wrapper.h"

/*******************************************************************************/


/* ===================================================================== */
/*
IMB 3.1 changes
July 2007
Hans-Joachim Plum, Intel GmbH

- replace "int n_sample" by iteration scheduling object "ITERATIONS"
  (see => IMB_benchmark.h)

- proceed with offsets in send / recv buffers to eventually provide
  out-of-cache data
*/
/* ===================================================================== */

void IMB_dcclallreduce(struct comm_info* c_info, int size, struct iter_schedule* ITERATIONS,
                   MODES RUN_MODE, double* time) {
/*

                      MPI-1 benchmark kernel
                      Benchmarks MPI_Allreduce


Input variables:

-c_info               (type struct comm_info*)
                      Collection of all base data for MPI;
                      see [1] for more information

-size                 (type int)
                      Basic message size in bytes

-ITERATIONS           (type struct iter_schedule *)
                      Repetition scheduling


-RUN_MODE             (type MODES)
                      (only MPI-2 case: see [1])


Output variables:

-time                 (type double*)
                      Timing result per sample


*/
    int    i;

    Type_Size s_size;
    int s_num = 0;
#ifdef CHECK
    int asize = (int) sizeof(assign_type);
    defect = 0.;
#endif

    *time = 0.;

    /*  GET SIZE OF DATA TYPE */
    MPI_Type_size(c_info->red_data_type, &s_size);
    if (s_size != 0)
        s_num = size / s_size;

    size *= c_info->size_scale;

    for (i = 0; i < ITERATIONS->n_sample; i++) {
        *time -= MPI_Wtime();
        // MPI_ERRHAND(MPI_Allreduce((char*)c_info->s_buffer + i % ITERATIONS->s_cache_iter * ITERATIONS->s_offs,
        //                             (char*)c_info->r_buffer + i % ITERATIONS->r_cache_iter * ITERATIONS->r_offs,
        //                             s_num,
        //                             c_info->red_data_type,c_info->op_type,
        //                             c_info->communicator));
        DCCLAllReduceWrapper::singleton.DCCL_Allreduce((char*)c_info->s_buffer + i % ITERATIONS->s_cache_iter * ITERATIONS->s_offs,
                                    (char*)c_info->r_buffer + i % ITERATIONS->r_cache_iter * ITERATIONS->r_offs,
                                    s_num);
        *time += MPI_Wtime();

        // CHK_DIFF("Allreduce",c_info, (char*)c_info->r_buffer + i % ITERATIONS->r_cache_iter * ITERATIONS->r_offs, 0,
        //          size, size, asize,
        //          put, 0, ITERATIONS->n_sample, i,
        //          -1, &defect);
    }
    *time /= ITERATIONS->n_sample;
}
