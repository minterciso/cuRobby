/*
 * =====================================================================================
 *
 *       Filename:  prng.cu
 *
 *    Description:  Implementation of PRNG generic methods to run on the GPU
 *
 *        Version:  1.0
 *        Created:  17/01/2019 17:33:12
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#include "prng.h"
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

#include "utils.h"

__global__ void setup_prng(curandState *state, unsigned long long seed, unsigned int amount){
    const int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < amount)
        curand_init(seed, 0, tid, &state[tid]);
}

__global__ void test_prng(curandState *state, unsigned int state_amnt, float *data, int data_amount){
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < state_amnt && tid < data_amount){
        float sum = 0.0;
        curandState local_state = state[tid];
        for(int i=0;i<100;i++)
            sum += curand_uniform(&local_state);
        data[tid] = sum/100;
        state[tid] = local_state;
    }
}

