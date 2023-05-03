#include "DCCL_allreduce_wrapper.h"
#include <dccl.hpp>
#include <iostream>

DCCLAllReduceWrapper DCCLAllReduceWrapper::singleton;

DCCLAllReduceWrapper::DCCLAllReduceWrapper() {
    ncclResult_t ret;

    // step 1 - initialize comm
    ret = ncclCommInit(&dccl_comm);

    if (ret != ncclSuccess) {
        std::cerr << "Unsuccessful dccl comm initialization." << std::endl;
    }
}

int DCCLAllReduceWrapper::DCCL_Allreduce(void* in, void* out, int count, MPI_Datatype type, MPI_Comm comm) {
    ncclResult_t ret;

    ret = ncclAllReduce(reinterpret_cast<const void*>(in), out, count, ncclUint32, ncclSum, dccl_comm);
    return ret;
}

DCCLAllReduceWrapper::~DCCLAllReduceWrapper() {
    ncclCommFinalize(dccl_comm);
}
