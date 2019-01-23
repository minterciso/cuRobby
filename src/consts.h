/*
 * =====================================================================================
 *
 *       Filename:  consts.h
 *
 *    Description:  Define the constants used on the GA and Robby
 *
 *        Version:  1.0
 *        Created:  17/01/2019 17:30:49
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#ifndef __CONSTS_H
#define __CONSTS_H

/*
 * Some definitions:
 * W_ => World
 * T_ => Tile
 * P_ => Probability
 */

// World
#define W_ROWS 10
#define W_COLS 10
#define T_EMPTY 0
#define T_CAN   1
#define P_CAN 0.5

// Robby
#define R_START_ROW 0
#define R_START_COL 0
// Robby strategies
#define S_MOVE_NORTH  0
#define S_MOVE_SOUTH  1
#define S_MOVE_EAST   2
#define S_MOVE_WEST   3
#define S_STAY_PUT    4
#define S_PICK_UP     5
#define S_RANDOM      6
#define S_MAX_OPTIONS 7
#define S_SIZE 243

#endif // __CONSTS_H
