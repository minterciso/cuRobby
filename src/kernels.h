/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: minterciso
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include "consts.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <gsl/gsl_rng.h>

static curandState *d_randState; // Store globally the states, to be used on other modules and kernels as well
static gsl_rng *h_prng;

typedef struct robby{
    int neighbours[5];
    int strategy[S_SIZE];
    float fitness;
    float weight;
} robby;

typedef struct {
  int tiles[W_ROWS][W_COLS];
  int r_row;
  int r_col;
  int qtd_cans;
}world;

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

/**
 * @brief Create all worlds in one execution of the kernel.
 * @param states An already initialized curandState array
 * @param amount_states How many states are there in the array
 * @param d_worlds A pointer to a device allocate memory of worlds
 * @param amount_world How many worlds are there
 * @note We need the amount_states to be bigger than the amount_world, otherwise we will get the same result as old states
 */
__global__ void create_worlds(curandState *states, int amount_states, world* d_worlds, int amount_world);

/**
 * @brief Actually creates the world. This function can also be used to reset the world to a new random configuration
 * @param state The random state to use
 * @param d_world The pointer to the world to create
 * @return The amount of cans on the world
 */
__device__ int create_world(curandState *state, world *d_world);


__global__ void create_population(curandState *states, int amount_states, robby *d_robby, int amount_robby);

__global__ void execute_population(curandState *states, int amount_states, robby *d_robby, int amount_robby/*, world *d_world*/, int amount_world);

__device__ int execute_strategy(curandState *state, robby *d_robby, world *w);


#endif /* KERNELS_H_ */
