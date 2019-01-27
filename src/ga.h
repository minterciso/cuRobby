/*
 * ga.h
 *
 *  Created on: 23/01/2019
 *      Author: minterciso
 */

#ifndef GA_H_
#define GA_H_

/**
 * @brief Execute the Genetic Algorithm to evolve a good strategy for Robby
 * @param selection_type The type of selecion
 * @param output The output file name with the evolution in a csv format
 */
void execute_ga(int selection_type, const char *output);


#endif /* GA_H_ */
