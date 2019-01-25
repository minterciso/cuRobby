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

__global__ void execute_population(curandState *states, int amount_states, robby *d_robby, int amount_robby, world *d_world, int amount_world){
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if(tid < amount_states && tid < amount_robby){
    int score = 0;
    for(int world_id = 0; world_id < amount_world; world_id++){
      for(int i=0;i<200;i++)
        score += execute_strategy(&states[tid], &d_robby[tid], &d_world[world_id]);
    }
    d_robby[tid].fitness = (float)(score/amount_world);
  }
}

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
