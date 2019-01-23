#include "utils.h"

void start_device(void){
    CUDA_CALL(cudaSetDevice(0));
}

void reset_device(void){
    CUDA_CALL(cudaDeviceReset());
}
