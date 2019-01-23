/*
 * =====================================================================================
 *
 *       Filename:  world.cu
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
#include "world.h"
#include "prng.h"

#include <stdio.h>

__global__ void create_worlds(curandState *states, int amount_states, world* d_worlds, int amount_worlds){
  const int state_id = threadIdx.x + blockIdx.x*blockDim.x;
  if(state_id < amount_states && state_id < amount_worlds){
    curandState local_state = states[state_id];
    for(int i=0;i<W_ROWS;i++){
      for(int j=0;j<W_COLS;j++){
        if(curand_uniform(&local_state) < P_CAN){
          d_worlds[state_id].tiles[i][j] = T_CAN;
          d_worlds[state_id].qtd_cans++;
        }
      }
    }
    states[state_id] = local_state;
  }
}

__device__ int create_world(curandState *state, world *d_world){
  return 0;
}

__device__ int reset_world(curandState *state, world *d_world){
  return 0;
}