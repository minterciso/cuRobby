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
