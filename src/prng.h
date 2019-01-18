/*
 * =====================================================================================
 *
 *       Filename:  prng.h
 *
 *    Description:  This module is used to create PRNG numbers on the device.
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
#ifndef __PRNG_H
#define __PRNG_H

#include <cuda.h>
#include <curand_kernel.h>

static curandState *d_randState; // Store globally the states, to be used on other modules and kernels as well

__global__ void setup_prng(curandState *state, unsigned long long seed, unsigned int amount);

__global__ void test_prng(curandState *state, unsigned int state_amnt, float *data, int data_amount);

__device__ int get_uniform(curandState *state, int min, int max);

__global__ void test_prng_uniform(curandState *states, unsigned int state_amnt, int *data, int data_amount, int min, int max);

__global__ void test_prng_uniform(curandState *states, unsigned int state_amnt, float *data, int data_amount);

#endif // __PRNG_H

