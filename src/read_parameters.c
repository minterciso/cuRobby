/*
 * read_parameters.c
 *
 *  Created on: 27/01/2019
 *      Author: minterciso
 */
#include "read_parameters.h"
#include "consts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ga_options* read_params(const char *param_file){
  ga_options *opt = NULL;
  size_t option_bytes = sizeof(ga_options);
  if((opt=(ga_options*)malloc(option_bytes))==NULL){
    perror("malloc");
    abort();
  }
  memset(opt, 0, option_bytes);

  if(param_file == NULL){
    // Get defaults
    opt->ga_pop_elite = GA_POP_ELITE;
    opt->ga_pop_size = GA_POP_SIZE;
    opt->ga_prob_mutation = GA_PROB_MUTATION;
    opt->ga_prob_xover = GA_PROB_XOVER;
    opt->ga_runs = GA_RUNS;
    opt->ga_tournament_amount = GA_TOURNAMENT_AMOUNT;
    opt->ga_worlds = GA_WORLDS;
    return opt;
  }
  return opt;
}
