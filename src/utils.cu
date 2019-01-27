#include "utils.h"
#include <sys/time.h>
#include <stdio.h>

void start_device(void){
  CUDA_CALL(cudaSetDevice(0));
}

void reset_device(void){
  CUDA_CALL(cudaDeviceReset());
}

double cpu_second(void){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
