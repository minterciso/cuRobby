/*
 * =====================================================================================
 *
 *       Filename:  test_device.h
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
#ifndef __TEST_DEVICE_H
#define __TEST_DEVICE_H

void test_prng(void);

void test_uniform_prng(int min, int max);

#endif //__TEST_DEVICE_H

