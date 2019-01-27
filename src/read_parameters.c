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
  FILE *fp = NULL;
  if((fp = fopen(param_file, "r"))==NULL){
    perror("fopen");
    abort();
  }
  char buf[1024];
  char option[80];
  char val[80];
  int idx = 0;
  memset(buf, 0, 1024);
  while(fgets(buf, 1024, fp)!=NULL){
    memset(option, 0, 80);
    memset(val, 0, 80);
    buf[strlen(buf)-1] = '\0';
    idx=0;
    for(int i=0;i<strlen(buf);i++){
      if(buf[i] == ' ')
        break;
      option[i] = buf[i];
      idx++;
    }
    snprintf(val, strlen(buf)-idx+1, "%s", &buf[idx]);
    if(strcmp(option, "GA_RUNS")==0)
      opt->ga_runs = atoi(val);
    else if(strcmp(option, "GA_TOURNAMENT_AMOUNT")==0)
      opt->ga_tournament_amount = atoi(val);
    else if(strcmp(option, "GA_POP_SIZE")==0)
      opt->ga_pop_size = atoi(val);
    else if(strcmp(option, "GA_POP_ELITE")==0)
      opt->ga_pop_elite = atoi(val);
    else if(strcmp(option, "GA_PROB_MUTATION")==0)
      opt->ga_prob_mutation = atof(val);
    else if(strcmp(option, "GA_PROB_XOVER")==0)
      opt->ga_prob_xover = atof(val);
    else if(strcmp(option, "GA_WORLDS")==0)
      opt->ga_worlds = atoi(val);
  }
  fclose(fp);
  return opt;
}
