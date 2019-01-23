/*
 * robby.cu
 *
 *  Created on: 23/01/2019
 *      Author: minterciso
 */
#include "robby.h"
#include "consts.h"
#include "prng.h"

#include <cuda.h>



__global__ void create_population(curandState *states, int amount_states, robby *d_robby, int amount_robby){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(tid < amount_states && tid < amount_robby){
    curandState local_state = states[tid];
    d_robby[tid].fitness = 0.0;
    d_robby[tid].weight = 0.0;
    for(int i=0;i<S_SIZE;i++)
      //d_robby[tid].strategy[i] = get_uniform(&local_state, 0, S_MAX_OPTIONS);
      d_robby[tid].strategy[i] = (int)(curand_uniform(&local_state) * S_MAX_OPTIONS);
    states[tid] = local_state;
  }
}

