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

void test_prng(void){
    int amount = 1024;

    fprintf(stdout,"[*] Testing PRNG functions\n");
#ifdef DEBUG
    fprintf(stdout,"[D] Calling start_prng()\n");
#endif
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
}
