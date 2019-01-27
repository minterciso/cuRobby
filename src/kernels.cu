/*
 * kernel.cu
 *
 *  Created on: 26/01/2019
 *      Author: minterciso
 */
#include "kernels.h"
#include "consts.h"
#include "utils.h"

#include <cuda.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>


/*********
 * ROBBY *
 ********/
/**
 * @brief Execute the population and get the fitness for every individual
 * This function will execute the population in parallel on the GPU, each thread will handle exactly 1 individual.
 * @param states An already initialized curandState array (one per thread)
 * @param amount_states The amount of states present on the array
 * @param d_robby The already initialized population
 * @param amount_world The amount of world for each individual to run
 * @note There are some small improvements that we can make here, for instance, if we are SURE that the amount of states is the amount of threads, there's no use for the if and
 * the amount_states and amount_robby parameters.
 */
__global__ void execute_population(curandState *states, int amount_states, robby *d_robby, int amount_robby, int amount_world){
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if(tid < amount_states && tid < amount_robby){
    for(int world_id = 0; world_id < amount_world; world_id++){
      int score = 0;
      world world;;
      create_world(&states[tid], &world);
      for(int i=0;i<200;i++)
        score += execute_strategy(&states[tid], &d_robby[tid], &world);
      d_robby[tid].fitness += (float)score;
    }
    d_robby[tid].fitness /= amount_world;
  }
}

/**
 * @brief Convert an array in any base to decimal
 * @param arr The array to convert
 * @param base The base the array currently is
 * @param len The size of the array
 * @return The decimal converted number
 * @warning Currently this has a caps limit for int numbers
 */
__device__ int to_decimal(int *arr, int base, int len){
  int power = 1;
  int num = 0;
  int i;
  for(i=len-1;i>=0;i--){
    if(arr[i] >= base)
      return -1;
    num += (arr[i] * power);
    power *= base;
  }
  return num;
}

/**
 * @brief Create the population on the GPU
 * @note Each thread on the GPU is responsible for creating one individual
 * @param states The curandState states (one per thread)
 * @param amount_states The amount of states on the array
 * @param d_robby The allocated population on the device
 * @param amount_robby The size of the population
 * @note There are some small improvements that we can make here, for instance, if we are SURE that the amount of states is the amount of threads, there's no use for the if and
 * the amount_states and amount_robby parameters.
 */
__global__ void create_population(curandState *states, int amount_states, robby *d_robby, int amount_robby){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(tid < amount_states && tid < amount_robby){
    curandState local_state = states[tid];
    d_robby[tid].fitness = -99.0;
    d_robby[tid].weight = -99.0;
    for(int i=0;i<S_SIZE;i++)
      d_robby[tid].strategy[i] = (int)(curand_uniform(&local_state) * S_MAX_OPTIONS);
    states[tid] = local_state;
  }
}

/**
 * @brief Execute the strategy and return the score
 * @param state The curandState to use for random movements
 * @param d_robby The individual we are executing
 * @param w The world we are facing
 * @return The score for one execution of the strategy
 */
__device__ int execute_strategy(curandState *state, robby *d_robby, world *w){
  int n[5]; // neighbours
  int r_row, r_col; // robby position
  int strategy_id;
  int strategy_step;
  int movement;
  int score = 0;

  // Get robby position
  r_row = w->r_row;
  r_col = w->r_col;

  // Get neighbours
  if(r_row == 0)
    n[0] = T_WALL;
  else
    n[0] = w->tiles[r_row-1][r_col];
  if(r_row == W_ROWS - 1)
    n[1] = T_WALL;
  else
    n[1] = w->tiles[r_row+1][r_col];
  if(r_col == 0)
    n[2] = T_WALL;
  else
    n[2] = w->tiles[r_row][r_col-1];
  if(r_col == W_COLS - 1)
    n[3] = T_WALL;
  else
    n[3] = w->tiles[r_row][r_col+1];
  n[4] = w->tiles[r_row][r_col];

  // Now get the id of the strategy
  strategy_id = to_decimal(n, 3, 5);
  strategy_step = d_robby->strategy[strategy_id];
  if(strategy_step == S_RANDOM)
    movement = curand_uniform(state) * 4;
  else
    movement = strategy_step;
  switch(movement){
    case S_MOVE_NORTH:
      if(r_row == 0)
        score = -5;
      else
        w->r_row--;
      break;
    case S_MOVE_SOUTH:
      if(r_row == W_ROWS-1)
        score = -5;
      else
        w->r_row++;
      break;
    case S_MOVE_WEST:
      if(r_col == 0)
        score = -5;
      else
        w->r_col--;
      break;
    case S_MOVE_EAST:
      if(r_col == W_COLS - 1)
        score = -5;
      else
        w->r_col++;
      break;
    case S_STAY_PUT:
      score = 0;
      break;
    case S_PICK_UP:
      if(w->tiles[r_row][r_col] == T_CAN){
          score = 10;
          w->tiles[r_row][r_col] = T_EMPTY;
          w->qtd_cans--;
      }
      else
        score = -1;
      break;
  }

  return score;
}

/********
 * PRNG *
 *******/
/**
 * @brief Start the device PRNG
 * @param state The allocated array of curandState to initialize
 * @param seed The seed to start each state
 * @param amount The amount of state to initialize
 * @note There are some small improvements that we can make here, for instance, if we are SURE that the amount of states is the amount of threads, there's no use for the if and
 * the amount parameter.
 */
__global__ void setup_prng(curandState *state, unsigned long long seed, unsigned int amount){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(tid < amount)
    curand_init(seed, 0, tid, &state[tid]);
}

/**
 * @brief Returns an uniform value between min and max
 * @param state The state to get the value from
 * @param min The minimum value
 * @param max The maximum value
 * @return A value between (min,max( from a uniform distribution
 */
__device__ int get_uniform(curandState *state, int min, int max){
  return (int)( min + curand_uniform(state) * max);
}

/*********
 * WORLD *
 ********/
/**
 * @brief Create one world with random cans to execute
 * @param state The curandState to use
 * @param d_world An initialized world to work on
 * @return The amount of cans
 */
__device__ int create_world(curandState *state, world *d_world){
  d_world->qtd_cans = 0;
  d_world->r_row = 0;
  d_world->r_col = 0;
  for(int i=0;i<W_ROWS;i++){
    for(int j=0;j<W_COLS;j++){
      if(curand_uniform(state) < P_CAN){
        d_world->tiles[i][j] = T_CAN;
        d_world->qtd_cans++;
      }
      else{
        d_world->tiles[i][j] = T_EMPTY;
      }
    }
  }
  return d_world->qtd_cans;
}
