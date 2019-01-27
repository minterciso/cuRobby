/*
 * ga.cu
 *
 *  Created on: 23/01/2019
 *      Author: minterciso
 */
#include "ga.h"
#include "kernels.h"
#include "utils.h"
#include "consts.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>

/**
 * @brief Compare individuals (used on quicksort)
 * @param p1 A pointer to the 1st individual
 * @param p2 A pointer to the 2nd individual
 * @return -1 if the 1st fitness is greater than the 2nd, 1 otherwise, or 0 if equals
 */
static int cmp_robby(const void *p1, const void *p2){
  robby *r1 = (robby*)p1;
  robby *r2 = (robby*)p2;
  if(r1->fitness > r2->fitness) return -1;
  if(r2->fitness > r1->fitness) return 1;
  return 0;
}

/**
 * @brief Starts the PRNG on the device
 * @param amount The amount of states to start
 */
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

/**
 * @brief Stop the PRNG on the device
 */
void stop_prng_device(void){
  fprintf(stdout,"[*] Clearing PRNG Device memory\n");
  fflush(stdout);
  CUDA_CALL(cudaFree(d_randState));
  fprintf(stdout,"[*] Done!\n");
  fflush(stdout);
}

/**
 * @brief Start the GSL PRNG on the host
 * @return -1 if error, 0 otherwise
 */
int start_host_prng(void){
  const gsl_rng_type *T = gsl_rng_mt19937;
  if((h_prng=gsl_rng_alloc(T))==NULL){
    fprintf(stderr,"[E] Error starting PRNG\n");
    return -1;
  }
  struct timeval tp;
  gettimeofday(&tp, (struct timezone *)0);
  gsl_rng_set(h_prng, tp.tv_usec);
  return 0;
}

/**
 * @brief Stop the PRNG on the Host
 */
void stop_host_prng(void){
  gsl_rng_free(h_prng);
}

/**
 * @brief Select an individual based on the selection_type parameter
 * @param pop The population to look for the individual
 * @param weighted_sum The summed weight used on the roulette selection
 * @param selection_type The selection type to search individuals
 */
int select_individual(robby *pop, double weighted_sum, int selection_type){
  if(selection_type == GA_SELECTION_ROULETTE){
    float rnd = 0.0;
    float sum = 0.0;
    int index = -1;

    // Now select a random value from 0 to weighted_sum
    rnd = gsl_rng_uniform(h_prng) * weighted_sum;
    // Now find the index corresponding to the rnd probability
    while(sum < rnd){
      index++;
      sum += pop[index].weight;
    }
    return index;
  }
  if(selection_type == GA_SELECTION_ELITE){
    return gsl_rng_uniform_int(h_prng, GA_POP_ELITE);
  }
  if(selection_type == GA_SELECTION_TOURNAMENT){
    // Select at random 2 individuals and return the one with the best fitness
    int idx1, idx2;
    idx1 = gsl_rng_uniform_int(h_prng, GA_POP_SIZE);
    idx2 = gsl_rng_uniform_int(h_prng, GA_POP_SIZE);
    if(pop[idx1].fitness > pop[idx2].fitness) return idx1;
    return idx2;
  }
  return -1;
}

/**
 * @brief Create a new population base on the selection_type
 * @param debugFP An already openned file for debug (defaults to output/debug/ directory)
 * @param pop The population (old, and will receive the new one)
 * @param selection_type The selection type to use
 */
int create_new_population(FILE *debugFP, robby *pop, int selection_type){
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
#ifdef DEBUG
  fprintf(debugFP, "Weighted Sum: %f\n", weighted_sum);
#endif
  for(int i=0;i<GA_POP_SIZE;i++){
    int p1_idx, p2_idx;
    int xp;
    // Select 2 parents
    p1_idx = select_individual(old_pop, weighted_sum, selection_type);
    p2_idx = select_individual(old_pop, weighted_sum, selection_type);
    // If we have to crossover...
    if(gsl_rng_uniform(h_prng) < GA_PROB_XOVER){
      // Select a random xover point
      xp = gsl_rng_uniform_int(h_prng, S_SIZE);
      // Create 1 son
      for(int j=0;j<xp;j++)
        pop[i].strategy[j] = old_pop[p1_idx].strategy[j];
      for(int j=xp;j<S_SIZE;j++)
        pop[i].strategy[j] = old_pop[p2_idx].strategy[j];
    }
    else // If we don't need to crossover, just copy the strategy
      memcpy(&pop[i].strategy, &old_pop[p1_idx].strategy, sizeof(int)*S_SIZE);
#ifdef DEBUG
    fprintf(debugFP, "----------------------\n");
    fprintf(debugFP, "Parent 1: %d (%.5f) [", p1_idx, old_pop[p1_idx].fitness);
    for(int k=0;k<S_SIZE;k++) fprintf(debugFP, "%d", old_pop[p1_idx].strategy[k]);
    fprintf(debugFP, "]\nParent 2: %d (%.5f) [", p2_idx, old_pop[p2_idx].fitness);
    for(int k=0;k<S_SIZE;k++) fprintf(debugFP, "%d", old_pop[p2_idx].strategy[k]);
    fprintf(debugFP, "]\n");
    fprintf(debugFP, "XP: %d\n", xp);
    fprintf(debugFP, "Strategy:");
    for(int k=0;k<xp;k++) fprintf(debugFP, "%d", old_pop[p1_idx].strategy[k]);
    fprintf(debugFP, "|");
    for(int k=xp;k<S_SIZE;k++) fprintf(debugFP, "%d", old_pop[p2_idx].strategy[k]);
    fprintf(debugFP, "\n");
    fprintf(debugFP, "Mutation:\n");
#endif
    pop[i].fitness = -99.99;
    pop[i].weight = -99.99;
    // Now check for mutation
    for(int j=0;j<S_SIZE;j++){
      if(gsl_rng_uniform(h_prng) < GA_PROB_MUTATION){
        pop[i].strategy[j] = gsl_rng_uniform_int(h_prng, S_MAX_OPTIONS);
#ifdef DEBUG
        fprintf(debugFP, "(%d) = %d\n", j,pop[i].strategy[j]);
#endif
      }
    }
  }
  free(old_pop);
  return 0;
}

void execute_ga(int selection_type, const char *output){
  FILE *debugFP = NULL;
#ifdef DEBUG
  char fname[30];
  memset(fname, 0, 30);
#endif
  //parameters
  int prng_amount = GA_POP_SIZE;
  robby *h_population, *d_population;
  world *d_worlds;
  size_t population_bytes = sizeof(robby)*GA_POP_SIZE;
  size_t world_bytes = sizeof(world)*GA_WORLDS;
  FILE *fp;
  fprintf(stdout,"[*] Creating file '%s'\n", output);
  if((fp = fopen(output, "w"))==NULL){
    perror("fopen");
    abort();
  }
  fprintf(fp,"generations, fitness\n");

  fprintf(stdout,"[*] Starting PRNG\n");
#ifdef DEBUG
  fprintf(stdout,"[D] - Host\n");
#endif

#ifdef DEBUG
  fprintf(stdout,"[D] - Device\n");
  fprintf(stdout,"[D] - # States: %d\n", prng_amount);
#endif
  start_prng_device(prng_amount);
  start_host_prng();

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
#ifdef DEBUG
    snprintf(fname, 30, "output/debug/xover_%d.log", g);
    if((debugFP = fopen(fname, "w"))==NULL){
      perror("fopen");
      abort();
    }
#endif
    num_blocks = GA_POP_SIZE/num_threads + 1;
    execute_population<<<num_blocks, num_threads>>>(d_randState, prng_amount, d_population, GA_POP_SIZE/*, d_worlds*/, GA_WORLDS);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaMemcpy(h_population, d_population, population_bytes, cudaMemcpyDeviceToHost));
    qsort(h_population, GA_POP_SIZE, sizeof(robby), cmp_robby);
    fprintf(fp,"%d,%.10f\n", g, h_population[0].fitness);
#ifdef DEBUG
    fprintf(stdout,"[D] %03d: %.2f\n", g, h_population[0].fitness);
    fprintf(stdout,"[D] %03d: ", g);
    for(int i=0;i<GA_POP_SIZE;i++) fprintf(stdout,"%.2f ", h_population[i].fitness);
    fprintf(stdout,"\n");
#endif
    if(g == GA_RUNS - 1)
      break;
    create_new_population(debugFP, h_population, selection_type);
    CUDA_CALL(cudaMemcpy(d_population, h_population, population_bytes, cudaMemcpyHostToDevice));
#ifdef DEBUG
    fclose(debugFP);
#endif
  }
  fprintf(stdout,"[*] Best:\n");
  fprintf(stdout,"[*] - Fitness: %.10f\n",h_population[0].fitness);
  fprintf(stdout,"[*] - Strategy: ");
  for(int i=0;i<S_SIZE;i++) fprintf(stdout,"%d", h_population[0].strategy[i]);
  fprintf(stdout,"\n");

  fprintf(stdout,"[*] Cleaning\n");
  stop_prng_device();
  stop_host_prng();
  free(h_population);
  CUDA_CALL(cudaFree(d_population));
  CUDA_CALL(cudaFree(d_worlds));
  fclose(fp);
}
