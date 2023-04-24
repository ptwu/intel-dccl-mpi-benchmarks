#include "DCCL_allreduce_wrapper.h"
#include <dccl.hpp>

ncclComm_t dccl_comm;

DCCLAllReduceWrapper::DCCLAllReduceWrapper() {
    ncclResult_t ret;

    // step 1 - initialize comm
    ret = ncclCommInit(&dccl_comm);
}

int DCCLAllReduceWrapper::DCCL_Allreduce(void* in, void* out, std::size_t count, MPI_Datatype type, MPI_Comm comm) {
    ncclResult_t ret;

    ret = ncclAllReduce(reinterpret_cast<const void*>(in), out, count, ncclUint32, ncclSum, dccl_comm);
    if (ret != ncclSuccess) {
        return ret;
    } 

    // step 3 - finalize comm
    ret = ncclCommFinalize(dccl_comm);
    return ret;
}