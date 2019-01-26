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
#include <gsl/gsl_rng.h>

//static curandState *d_randState; // Store globally the states, to be used on other modules and kernels as well
//static gsl_rng *h_prng;

/**
 * @brief Starts the PRNG on the GPU with default values
 * @param state An array of curandState allocated on the GPU
 * @param seed A seed that should be different for each execution
 * @param amount The size of the state array
 */
__global__ void setup_prng(curandState *state, unsigned long long seed, unsigned int amount);

__global__ void test_prng(curandState *state, unsigned int state_amnt, float *data, int data_amount);

/**
 * @brief Returns an int in the range of (min,max(, of a uniform distribution
 * @param state The curandState to use
 * @param min The minimum value possible
 * @param max The maximum value possible
 */
__device__ int get_uniform(curandState *state, int min, int max);

/**
 * @brief Returns a float from the uniform distribution within the range of (0,1(
 * @param state The curandState to use
 */
__device__ float get_uniform(curandState *state);

__global__ void test_prng_uniform(curandState *states, unsigned int state_amnt, int *data, int data_amount, int min, int max);

__global__ void test_prng_uniform(curandState *states, unsigned int state_amnt, float *data, int data_amount);

int start_host_prng(void);
void stop_host_prng(void);

#endif // __PRNG_H

