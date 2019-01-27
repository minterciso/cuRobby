
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "consts.h"
#include "utils.h"
#include "ga.h"
#include "read_parameters.h"

const char *prog_name;

void usage(FILE *stream, int exit_code){
  fprintf(stream, "Usage: %s options\n", prog_name);
  fprintf(stream, "\t-h\t\t--help\t\t\tThis help message\n");
  fprintf(stream, "\t-o file\t\t--output file\t\tEvolution output in a csv format\n");
  fprintf(stream, "\t-s type\t\t--selection type\tThe type of selection ('roulette', 'elite', 'tournament')\n");
  fprintf(stream, "\t-p file\t\t--parameter file\tThe parameter file to read from\n");
  fprintf(stream, "Defaults: -o output.csv -s roulette\n");
  exit(exit_code);
}

int main(int argc, char **argv)
{
  // Program options
  prog_name = argv[0];
  const char* const short_options = "ho:s:p:";
  const struct option long_options[] = {
      {"help",      0, NULL, 'h'},
      {"output",    1, NULL, 'o'},
      {"selection", 1, NULL, 's'},
      {"parameter", 1, NULL, 'p'},
      {0,0,0,0}
  };
  int next_option;
  int selection_type = GA_SELECTION_ROULETTE;
  char *output_fname = NULL;
  char *parameter_fname = NULL;
  size_t fname_bytes = 0, p_fname_bytes = 0;
  do{
    next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    switch(next_option){
      case 'h': usage(stdout,EXIT_SUCCESS); break;
      case 'o':
        fname_bytes = sizeof(char)*(strlen(optarg)+1);
        if((output_fname = (char*)malloc(fname_bytes))==NULL){
          perror("malloc");
          abort();
        }
        memset(output_fname, 0, fname_bytes);
        snprintf(output_fname, fname_bytes, "%s", optarg);
        break;
      case 'p':
        p_fname_bytes = sizeof(char)*(strlen(optarg)+1);
        if((parameter_fname=(char*)malloc(p_fname_bytes))==NULL){
          perror("malloc");
          abort();
        }
        memset(parameter_fname, 0, p_fname_bytes);
        snprintf(parameter_fname, p_fname_bytes, "%s", optarg);
        break;
      case 's':
        if(strcmp(optarg, "roulette") == 0) selection_type = GA_SELECTION_ROULETTE;
        else if(strcmp(optarg, "elite") == 0) selection_type = GA_SELECTION_ELITE;
        else if(strcmp(optarg, "tournament") == 0) selection_type = GA_SELECTION_TOURNAMENT;
        break;
      case '?': usage(stderr,EXIT_FAILURE); break;
      case -1: break;
      default: abort();
    }
  }while(next_option != -1);
  if(output_fname == NULL){
    fname_bytes = sizeof(char)*(strlen("output.csv")+1);
    if((output_fname=(char*)malloc(fname_bytes))==NULL){
      perror("malloc");
      abort();
    }
    memset(output_fname, 0, fname_bytes);
    snprintf(output_fname, fname_bytes, "output.csv");
  }
  ga_options *options = read_params(parameter_fname);

  fprintf(stdout,"[*] Parameters used:\n");
  fprintf(stdout,"[*] - Paramter file:         %s\n", (parameter_fname==NULL)?"None":parameter_fname);
  fprintf(stdout,"[*] - Generations:           %d\n", options->ga_runs);
  fprintf(stdout,"[*] - Population size:       %d\n", options->ga_pop_size);
  fprintf(stdout,"[*] - Elite size:            %d\n", options->ga_pop_elite);
  fprintf(stdout,"[*] - Tournament size:       %d\n", options->ga_tournament_amount);
  fprintf(stdout,"[*] - Amount of worlds:      %d\n", options->ga_worlds);
  fprintf(stdout,"[*] - Selection Type:        '");
  if(selection_type == GA_SELECTION_ROULETTE) fprintf(stdout,"roulette'\n");
  else if(selection_type == GA_SELECTION_ELITE) fprintf(stdout, "elite'\n");
  else if(selection_type == GA_SELECTION_TOURNAMENT) fprintf(stdout,"tournament'\n");
  fprintf(stdout,"[*] - Crossover probability: %.5f\n", options->ga_prob_xover);
  fprintf(stdout,"[*] - Mutation probability:  %.5f\n", options->ga_prob_mutation);
  fprintf(stdout,"[*] - Output file:           '%s'\n", output_fname);
#ifdef DEBUG
  fprintf(stdout,"[*] - Debug enabled!!!\n");
#endif
  fprintf(stdout,"[*] Starting device\n");
  start_device();

  fprintf(stdout,"[*] Starting evolution...\n");
  double t_start;
  t_start = cpu_second();
  execute_ga(selection_type, output_fname, options);
  fprintf(stdout,"[*] Finished in %f s\n", cpu_second() - t_start);

  free(output_fname);
  if(parameter_fname)
    free(parameter_fname);
  free(options);

  fprintf(stdout,"[*] Stopping device\n");
  reset_device();

  return EXIT_SUCCESS;
}
