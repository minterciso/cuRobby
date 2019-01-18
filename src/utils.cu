#include "utils.h"

int start_device(void){
    CUDA_CALL(cudaSetDevice(0));
    return 0;
}

int reset_device(void){
    CUDA_CALL(cudaDeviceReset());
    return 0;
}