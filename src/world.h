/*
 * =====================================================================================
 *
 *       Filename:  world.h
 *
 *    Description:  This defines the world of Robby. It needs to:
 *    - Describe the world for Robby
 *    - One Kernel to create a lot of worlds on the Device
 *    - One device function to create one world on the Device
 *
 *        Version:  1.0
 *        Created:  17/01/2019 17:30:49
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#ifndef __WORLD_H
#define __WORLD_H

#include <cuda.h>
#include <curand_kernel.h>

#include "consts.h"

typedef struct {
  int tiles[W_ROWS][W_COLS];
  int r_row;
  int r_col;
  int qtd_cans;
}world;

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

#endif // __WORLD_H

