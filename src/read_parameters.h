/*
 * read_parameters.h
 *
 *  Created on: 27/01/2019
 *      Author: minterciso
 */

#ifndef READ_PARAMETERS_H_
#define READ_PARAMETERS_H_

typedef struct {
    int ga_runs;
    int ga_pop_size;
    int ga_pop_elite;
    int ga_tournament_amount;
    int ga_worlds;
    float ga_prob_xover;
    float ga_prob_mutation;
} ga_options;

ga_options* read_params(const char *param_file);


#endif /* READ_PARAMETERS_H_ */
