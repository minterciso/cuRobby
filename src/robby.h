/*
 * robby.h
 *
 *  Created on: 23/01/2019
 *      Author: minterciso
 */

#ifndef ROBBY_H_
#define ROBBY_H_

#include "world.h"

#include <cuda.h>
#include <curand_kernel.h>

typedef struct robby{
    int neighbours[5];
    int strategy[S_SIZE];
    float fitness;
    float weight;
} robby;


__global__ void create_population(curandState *states, int amount_states, robby *d_robby, int amount_robby);

#endif /* ROBBY_H_ */
