/*
 * ga.cu
 *
 *  Created on: 23/01/2019
 *      Author: minterciso
 */
#include "ga.h"
#include "world.h"
#include "robby.h"
#include "consts.h"
#include "prng.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

static int cmp_robby(const void *p1, const void *p2){
  robby *r1 = (robby*)p1;
  robby *r2 = (robby*)p2;
  if(r1->fitness > r2->fitness) return -1;
  if(r2->fitness > r1->fitness) return 1;
  return 0;
}

void start_prng_device(int amount){
  size_t rand_bytes = sizeof(curandState)*amount;
  int numThreads = 512;
  int numBlocks = amount/numThreads + 1;

  fprintf(stdout,"[*] Starting PRNG\n");
  fflush(stdout);
#ifdef DEBUG
  fprintf(stdout,"[D] - Amount of states:  %d\n", amount);
  fprintf(stdout,"[D] - Amount of blocks:  %d\n", numBlocks);
  fprintf(stdout,"[D] - Amount of threads: %d\n", numThreads);
  fflush(stdout);
#endif

  CUDA_CALL(cudaMalloc((void**)&d_randState, rand_bytes));
  setup_prng<<<numBlocks, numThreads>>>(d_randState, time(NULL), amount);
  CUDA_CALL(cudaDeviceSynchronize());
  fprintf(stdout,"[*] Done!\n");
}

void stop_prng_device(void){
  fprintf(stdout,"[*] Clearing PRNG Device memory\n");
  fflush(stdout);
  CUDA_CALL(cudaFree(d_randState));
  fprintf(stdout,"[*] Done!\n");
  fflush(stdout);
}


int select_individual(robby *pop, double weighted_sum, int selection_type){
  if(selection_type == GA_SELECTION_ROULETTE){
    float rnd = 0.0;
    float sum = 0.0;
    int index = -1;

    // Now select a random value from 0 to weighted_sum
    rnd = (float)(rand()/(float)RAND_MAX) * weighted_sum;
    // Now find the index corresponding to the rnd probability
    while(sum < rnd){
      index++;
      sum += pop[index].weight;
    }
    return index;
  }
  if(selection_type == GA_SELECTION_ELITE){
    return rand() % GA_POP_ELITE;
  }
  if(selection_type == GA_SELECTION_TOURNAMENT){
    // Select at random 2 individuals and return the one with the best fitness
    int idx1, idx2;
    idx1 = rand() % GA_POP_SIZE;
    idx2 = rand() % GA_POP_SIZE;
    if(pop[idx1].fitness > pop[idx2].fitness) return idx1;
    return idx2;
  }
  return -1;
}

int crossover_and_mutate(robby *pop, int selection_type){
  robby *old_pop = NULL;
  size_t pop_bytes = sizeof(robby) * GA_POP_SIZE;

  // Calculate the weighted_sum
  double weighted_sum = 0;
  if(selection_type == GA_SELECTION_ROULETTE){
    // We rank the weights based on the sorting of the individual, due to the negative fitness values.
    for(int i=0;i<GA_POP_SIZE;i++){
      pop[i].weight = (float)(GA_POP_SIZE-i);
      weighted_sum += pop[i].weight;
    }
  }

  // Allocate memory for the old population
  if((old_pop=(robby*)malloc(pop_bytes))==NULL){
    perror("malloc");
    return -1;
  }
  memcpy(old_pop, pop, pop_bytes); // Copy the current population on the old population
  memset(pop, 0, pop_bytes); // Clear the current population

  // Now create a new population
  float rnd = 0.0;
  for(int i=0;i<GA_POP_SIZE;i++){
    int p1_idx, p2_idx;
    int xp;
    // Select 2 parents
    p1_idx = select_individual(old_pop, weighted_sum, selection_type);
    p2_idx = select_individual(old_pop, weighted_sum, selection_type);
    // If we have to crossover...
    rnd = (float)(rand()/(float)RAND_MAX);
    if(rnd < GA_PROB_XOVER){
      // Select a random xover point
      xp = rand() % S_SIZE;
      // Create 1 sons
      for(int j=0;j<xp;j++)
        pop[i].strategy[j] = old_pop[p1_idx].strategy[j];
      for(int j=xp;j<S_SIZE;j++)
        pop[i].strategy[j] = old_pop[p2_idx].strategy[j];
    }
    else // If we don't need to crossover, just copy the strategy
      memcpy(&pop[i].strategy, &old_pop[p1_idx].strategy, sizeof(int)*S_SIZE);
    pop[i].fitness = -99.99;
    pop[i].weight = -99.99;
    // Now check for mutation
    for(int j=0;j<S_SIZE;j++){
      rnd = (float)(rand()/(float)RAND_MAX);
      if(rnd < GA_PROB_MUTATION)
        pop[i].strategy[j] = rand() % S_MAX_OPTIONS;
    }
  }
  free(old_pop);
  return 0;
}

void execute_ga(void){
  /*
   * need to:
   * 1) start prng (device and host) (Done)
   * 2) create initial population (done)
   * 3) enter evolution loop (done)
   * 3a) create worlds (done)
   * 3b) execute_worlds (done)
   * 3c) copy back results (done)
   * 3d) sort population (done)
   * == if not last execution ==
   * 3e) create new population (done)
   * 3f) copy population to device (done)
   * 4) print result
   * Extra) store fitness on file
   * Extra1) create graph with gnuplot automatically
   */
  fprintf(stdout,"[*] Starting PRNG\n");
#ifdef DEBUG
  fprintf(stdout,"[D] - Host\n");
#endif
  //parameters
  int prng_amount = GA_POP_SIZE;
  robby *h_population, *d_population;
  world *d_worlds;
  size_t population_bytes = sizeof(robby)*GA_POP_SIZE;
  size_t world_bytes = sizeof(world)*GA_WORLDS;

  srand(time(NULL));
#ifdef DEBUG
  fprintf(stdout,"[D] - Device\n");
  fprintf(stdout,"[D] - # States: %d\n", prng_amount);
#endif
  start_prng_device(prng_amount);

  fprintf(stdout, "[*] Allocating memory\n");
#ifdef DEBUG
  fprintf(stdout,"[D] Population size: %.2f MB\n", (float)(population_bytes/(1024.0f*1024.0f)));
  fprintf(stdout,"[D] World size:      %.2f MB\n", (float)(world_bytes/(1024.0f*1024.0f)));
  fprintf(stdout,"[D] Host memory\n");
#endif
  if((h_population=(robby*)calloc(GA_POP_SIZE, sizeof(robby)))==NULL){
    perror("calloc");
    abort();
  }

#ifdef DEBUG
  fprintf(stdout,"[D] Device memory\n");
#endif
  CUDA_CALL(cudaMalloc((void**)&d_population, population_bytes));
  CUDA_CALL(cudaMalloc((void**)&d_worlds, world_bytes));
  CUDA_CALL(cudaMemset(d_population, 0, population_bytes));
  CUDA_CALL(cudaMemset(d_worlds, 0, world_bytes));

  fprintf(stdout,"[*] Creating initial population\n");
  int num_threads = 32;
  int num_blocks = GA_POP_SIZE/num_threads + 1;
#ifdef DEBUG
  fprintf(stdout,"[D] Kernel parameters:\n");
  fprintf(stdout," - # blocks:  %d\n", num_blocks);
  fprintf(stdout," - # threads: %d\n", num_threads);
#endif
  create_population<<<num_blocks, num_threads>>>(d_randState, prng_amount, d_population, GA_POP_SIZE);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaGetLastError());

  fprintf(stdout,"[*] Evolving\n");
  for(int g=0;g<GA_RUNS;g++){
    num_blocks = GA_WORLDS/num_threads + 1;
    create_worlds<<<num_blocks, num_threads>>>(d_randState, prng_amount, d_worlds, GA_WORLDS);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaGetLastError());
    num_blocks = GA_POP_SIZE/num_threads + 1;
    execute_population<<<num_blocks, num_threads>>>(d_randState, prng_amount, d_population, GA_POP_SIZE, d_worlds, GA_WORLDS);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaMemcpy(h_population, d_population, population_bytes, cudaMemcpyDeviceToHost));
    qsort(h_population, GA_POP_SIZE, sizeof(robby), cmp_robby);
    fprintf(stdout,"[*] %03d: %.2f\n", g, h_population[0].fitness);
//#ifdef DEBUG
    fprintf(stdout,"[D] %03d: ", g);
    for(int i=0;i<GA_POP_SIZE;i++) fprintf(stdout,"%.2f ", h_population[i].fitness);
    fprintf(stdout,"\n");
//#endif
    if(g == GA_RUNS - 1)
      break;
    crossover_and_mutate(h_population, GA_SELECTION);
    CUDA_CALL(cudaMemcpy(d_population, h_population, population_bytes, cudaMemcpyHostToDevice));
  }
  fprintf(stdout,"[*] Best:\n");
  fprintf(stdout,"[*] - Fitness: %.10f\n",h_population[0].fitness);
  fprintf(stdout,"[*] - Strategy: ");
  for(int i=0;i<S_SIZE;i++) fprintf(stdout,"%d", h_population[0].strategy[i]);
  fprintf(stdout,"\n");

  fprintf(stdout,"[*] Cleaning\n");
  stop_prng_device();
  free(h_population);
  CUDA_CALL(cudaFree(d_population));
  CUDA_CALL(cudaFree(d_worlds));
}
