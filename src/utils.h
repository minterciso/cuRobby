/*
 * =====================================================================================
 *
 *       Filename:  utils.h
 *
 *    Description:  Utilities for both device and host code
 *
 *        Version:  1.0
 *        Created:  17/01/2019 17:47:47
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <cuda.h>
#include "../config.h"

#define CUDA_CALL(call)                                                                 \
{                                                                                       \
        const cudaError_t error=call;                                                   \
        if(error != cudaSuccess){                                                       \
                printf("Error: %s:%d, ", __FILE__, __LINE__);                           \
                printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
                exit(1);                                                                \
                }                                                                       \
}

/**
 * @brief Start the first CUDA Device
 */
void start_device(void);

/**
 * @brief Resets the device for usage
 */
void reset_device(void);

/**
 * @brief Get's the cpu seconds and returns as a double
 * @return The amount of milliseconds since epoch
 */
double cpu_second(void);

#endif //__UTILS_H

