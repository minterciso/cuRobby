/*
 * =====================================================================================
 *
 *       Filename:  test_device.cu
 *
 *    Description:  Test kernels and methods on the device
 *
 *        Version:  1.0
 *        Created:  17/01/2019 17:43:23
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#include "test_device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "prng.h"
#include "consts.h"
#include "world.h"

void start_prng(int amount){
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

void stop_prng(void){
  fprintf(stdout,"[*] Clearing PRNG Device memory\n");
  fflush(stdout);
  CUDA_CALL(cudaFree(d_randState));
  fprintf(stdout,"[*] Done!\n");
  fflush(stdout);
}

void test_uniform_prng(int min, int max){
  int amount = 1024;
  fprintf(stdout,"[*] Testing Uniform PRNG function\n");
  float *h_f_data, *d_f_data;
  int *h_i_data, *d_i_data;
  int data_amount = 100;
  size_t f_data_bytes = sizeof(float)*data_amount;
  size_t i_data_bytes = sizeof(float)*data_amount;
  int num_threads = 32;
  int num_blocks = data_amount/num_threads + 1;

  fprintf(stdout,"[*] Allocating data arrays\n");
#ifdef DEBUG
  fprintf(stdout,"[D] Allocating host arrays\n");
#endif
  if((h_i_data = (int*)malloc(i_data_bytes))==NULL){
    perror("malloc");
    abort();
  }
  if((h_f_data = (float*)malloc(f_data_bytes))==NULL){
    perror("malloc");
    abort();
  }
  memset(h_i_data, 0, i_data_bytes);
  memset(h_f_data, 0, f_data_bytes);
#ifdef DEBUG
  fprintf(stdout,"[D] Allocating device array\n");
#endif
  CUDA_CALL(cudaMalloc((void**)&d_f_data, f_data_bytes));
  CUDA_CALL(cudaMalloc((void**)&d_i_data, i_data_bytes));
  CUDA_CALL(cudaMemset(d_f_data, 0, f_data_bytes));
  CUDA_CALL(cudaMemset(d_i_data, 0, i_data_bytes));
#ifdef DEBUG
  fprintf(stdout,"[D] Calling start_prrng()\n");
#endif
  start_prng(amount);

  fprintf(stdout,"[*] Creating random uniform numbers (0->1)\n");
#ifdef DEBUG
  fprintf(stdout,"[D] Kernel parameters:\n");
  fprintf(stdout,"[D] - Amount of Blocks:  %d\n", num_blocks);
  fprintf(stdout,"[D] - Amount of Threads: %d\n", num_threads);
#endif
  test_prng_uniform<<<num_blocks, num_threads>>>(d_randState, amount, d_f_data, data_amount);
  CUDA_CALL(cudaMemcpy(h_f_data, d_f_data, f_data_bytes, cudaMemcpyDeviceToHost));

  fprintf(stdout, "[*] Creating random uniform numbers (%d -> %d)\n", min,max);
  test_prng_uniform<<<num_blocks, num_threads>>>(d_randState, amount, d_i_data, data_amount, min, max);
  CUDA_CALL(cudaMemcpy(h_i_data, d_i_data, i_data_bytes, cudaMemcpyDeviceToHost));

  fprintf(stdout, "[*] Validating data\n");
  int valid = 0;
#ifdef DEBUG
  fprintf(stdout,"[D] Float data:");
#endif
  for(int i=0;i<data_amount;i++){
#ifdef DEBUG
    fprintf(stdout, "%.2f ", h_f_data[i]);
#endif
    if(h_f_data[i] > 0){
      valid=1;
#ifndef DEBUG
      break;
#endif
    }
  }
#ifdef DEBUG
  fprintf(stdout,"\n");
#endif
  if(valid==0)
    fprintf(stderr,"[E] Float data is invalid!\n");
  valid=0;
#ifdef DEBUG
  fprintf(stdout,"[D] Int data:");
#endif
  for(int i=0;i<data_amount;i++){
#ifdef DEBUG
    fprintf(stdout, "%d ", h_i_data[i]);
#endif
    if(h_i_data[i] > 0){
      valid=1;
#ifndef DEBUG
      break;
#endif
    }
  }
#ifdef DEBUG
  fprintf(stdout,"\n");
#endif
  if(valid == 0)
    fprintf(stderr,"[E] Int data is invalid!\n");
  else
    fprintf(stdout,"[*] Data is valid!\n");
  fprintf(stdout,"[*] Stopping PRNG\n");
  stop_prng();
  fprintf(stdout,"[*] Clearing data\n");
  CUDA_CALL(cudaFree(d_f_data));
  CUDA_CALL(cudaFree(d_i_data));
  free(h_f_data);
  free(h_i_data);
}

void test_prng(void){
  int amount = 1024;

  fprintf(stdout,"[*] Testing PRNG functions\n");
  float *h_data, *d_data;
  int data_amount = 100;
  size_t data_bytes = sizeof(float)*data_amount;
  int num_threads = 32;
  int num_blocks = data_amount/num_threads + 1;

  fprintf(stdout,"[*] Allocating data arrays\n");
#ifdef DEBUG
  fprintf(stdout,"[D] Allocating host array\n");
#endif
  if((h_data=(float*)malloc(data_bytes))==NULL){
    perror("malloc");
    abort();
  }
  memset(h_data, 0, data_bytes);
#ifdef DEBUG
  fprintf(stdout,"[D] Allocating device array\n");
#endif
  CUDA_CALL(cudaMalloc((void**)&d_data, data_bytes));
  CUDA_CALL(cudaMemset(d_data, 0, data_bytes));

#ifdef DEBUG
  fprintf(stdout,"[D] Calling start_prng()\n");
#endif
  start_prng(amount);
    
  fprintf(stdout,"[*] Creating random numbers\n");
#ifdef DEBUG
  fprintf(stdout,"[D] Starting kernel to find mean of 100 numbers, and store on an array with %d size\n", data_amount);
  fprintf(stdout,"[D] - Amount of blocks:  %d\n", num_blocks);
  fprintf(stdout,"[D] - Amount of threads: %d\n", num_threads);
#endif
  test_prng<<<num_blocks, num_threads>>>(d_randState, amount, d_data, data_amount);
  CUDA_CALL(cudaDeviceSynchronize());
#ifdef DEBUG
  fprintf(stdout,"[D] Getting memory from device to host\n");
#endif
  CUDA_CALL(cudaMemcpy(h_data, d_data, data_bytes, cudaMemcpyDeviceToHost));
  fprintf(stdout,"[*] Done, validating\n");
  int valid = 0;
#ifdef DEBUG
  fprintf(stdout,"[D] Values: ");
#endif
  for(int i=0;i<data_amount;i++){
#ifdef DEBUG
    fprintf(stdout, "%.2f ", h_data[i]);
#endif
    if(h_data[i] != 0.0){
      valid = 1;
#ifndef DEBUG
      break;
#endif
    }
  }
#ifdef DEBUG
  fprintf(stdout,"\n");
#endif
  if(valid == 1)
    fprintf(stdout,"[*] Data is valid!\n");
  else
    fprintf(stdout,"[W] Data is 0!\n");
  fprintf(stdout,"[*] Stopping PRNG\n");
  stop_prng();
  fprintf(stdout,"[*] Clearing data\n");
  CUDA_CALL(cudaFree(d_data));
  free(h_data);
}

void test_world_creation(int amount){
  fprintf(stdout,"[*] Starting World Creation...\n");

  int prng_amount = amount;
  world *h_worlds, *d_worlds;
  size_t world_bytes = sizeof(world)*amount;

  #ifdef DEBUG
  fprintf(stdout,"[D] Starting PRNG\n");
  fprintf(stdout,"[D] - Using %d states\n", prng_amount);
  #endif
  start_prng(prng_amount);

  fprintf(stdout,"[*] Allocating memory\n");
  #ifdef DEBUG
  fprintf(stdout,"[D] - Amount: %d\n", amount);
  fprintf(stdout,"[D] - MB:     %.2f\n", (float)world_bytes/(1024.0f*1024.0f));
  fprintf(stdout,"[D] Allocating host memory\n");
  #endif
  if((h_worlds=(world*)malloc(world_bytes))==NULL){
    perror("malloc");
    abort();
  }
  memset(h_worlds, 0, world_bytes);
  #ifdef DEBUG
  fprintf(stdout,"[D] Allocating Device memory\n");
  #endif
  CUDA_CALL(cudaMalloc((void**)&d_worlds, world_bytes));
  CUDA_CALL(cudaMemset(d_worlds, 0, world_bytes));

  fprintf(stdout,"[*] Calling kernel to create all worlds at once\n");
  /*
  dim3 num_threads(32,1);
  dim3 num_blocks( (prng_amount/num_threads.x + 1), 1);
  */
  dim3 num_threads(512);
  dim3 num_blocks(amount/num_threads.x + 1);
  #ifdef DEBUG
  fprintf(stdout,"[D] Kernel parameters:\n");
  fprintf(stdout,"[D] Num Blocks:  (%d, %d)\n",num_blocks.x, num_blocks.y);
  fprintf(stdout,"[D] Num Threads: (%d, %d)\n", num_threads.x, num_threads.y);
  #endif
  create_worlds<<<num_blocks, num_threads>>>(d_randState, prng_amount, d_worlds, amount);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaMemcpy(h_worlds, d_worlds, world_bytes, cudaMemcpyDeviceToHost));

  fprintf(stdout,"[*] Printing sample world 1: \n");
  world *w = &h_worlds[0];
  for(int i=0;i<W_ROWS; i++){
    for(int j=0;j<W_COLS;j++)
      fprintf(stdout,"%c", (w->tiles[i][j] == T_CAN)?'*':'_');
    fprintf(stdout,"\n");
  }
  fprintf(stdout,"[*] Amount of cans: %d\n", w->qtd_cans);
  fprintf(stdout,"[*] Printing sample world 2: \n");
  w = &h_worlds[1];
  for(int i=0;i<W_ROWS; i++){
    for(int j=0;j<W_COLS;j++)
      fprintf(stdout,"%c", (w->tiles[i][j] == T_CAN)?'*':'_');
    fprintf(stdout,"\n");
  }
  fprintf(stdout,"[*] Amount of cans: %d\n", w->qtd_cans);
  fprintf(stdout,"[*] Printing sample world %d: \n", amount-1);
  w = &h_worlds[amount-1];
  for(int i=0;i<W_ROWS; i++){
      for(int j=0;j<W_COLS;j++)
        fprintf(stdout,"%c", (w->tiles[i][j] == T_CAN)?'*':'_');
      fprintf(stdout,"\n");
    }
    fprintf(stdout,"[*] Amount of cans: %d\n", w->qtd_cans);
  fprintf(stdout,"[*] Cleaning up\n");
  stop_prng();
  free(h_worlds);
  CUDA_CALL(cudaFree(d_worlds));
}
