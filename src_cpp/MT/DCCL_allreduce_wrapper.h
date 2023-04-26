#pragma once
#include <dccl.hpp>
#include <mpi.h>

class DCCLAllReduceWrapper {
    private:
        ncclComm_t dccl_comm;

    public:
        DCCLAllReduceWrapper(); 
        // Assumes we are using MPI_Op: MPI_SUM.
        int DCCL_Allreduce(void* in, void* out, std::size_t count, MPI_Datatype type, MPI_Comm comm);
        virtual ~DCCLAllReduceWrapper();
        
        static DCCLAllReduceWrapper singleton;
};