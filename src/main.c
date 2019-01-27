
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "consts.h"
#include "utils.h"
#include "ga.h"

const char *prog_name;

void usage(FILE *stream, int exit_code){
  fprintf(stream, "Usage: %s options\n", prog_name);
  fprintf(stream, "\t-h\t\t--help\t\t\tThis help message\n");
  fprintf(stream, "\t-o file\t\t--output file\t\tEvolution output in a csv format\n");
  fprintf(stream, "\t-s type\t\t--selection type\tThe type of selection ('roulette', 'elite', 'tournament')\n");
  fprintf(stream, "Defaults: -o output.csv -s roulette\n");
  exit(exit_code);
}

int main(int argc, char **argv)
{
  // Program options
  prog_name = argv[0];
  const char* const short_options = "ho:s:";
  const struct option long_options[] = {
      {"help",      0, NULL, 'h'},
      {"output",    1, NULL, 'o'},
      {"selection", 1, NULL, 's'},
      {0,0,0,0}
  };
  int next_option;
  int selection_type = GA_SELECTION_ROULETTE;
  char *output_fname = NULL;
  size_t fname_bytes = 0;
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

  fprintf(stdout,"[*] Parameters used:\n");
  fprintf(stdout,"[*] - Generations:           %d\n", GA_RUNS);
  fprintf(stdout,"[*] - Population size:       %d\n", GA_POP_SIZE);
  fprintf(stdout,"[*] - Elite size:            %d\n", (int)(GA_POP_ELITE));
  fprintf(stdout,"[*] - Tournament size:       %d\n", GA_TOURNAMENT_AMOUNT);
  fprintf(stdout,"[*] - Amount of worlds:      %d\n", GA_WORLDS);
  fprintf(stdout,"[*] - Selection Type:        '");
  if(selection_type == GA_SELECTION_ROULETTE) fprintf(stdout,"roulette'\n");
  else if(selection_type == GA_SELECTION_ELITE) fprintf(stdout, "elite'\n");
  else if(selection_type == GA_SELECTION_TOURNAMENT) fprintf(stdout,"tournament'\n");
  fprintf(stdout,"[*] - Crossover probability: %.2f\n", GA_PROB_XOVER);
  fprintf(stdout,"[*] - Mutation probability:  %.2f\n", GA_PROB_MUTATION);
  fprintf(stdout,"[*] - Output file:           '%s'\n", output_fname);
#ifdef DEBUG
  fprintf(stdout,"[*] - Debug enabled!!!\n");
#endif
  fprintf(stdout,"[*] Starting device\n");
  start_device();

  fprintf(stdout,"[*] Starting evolution...\n");
  double t_start;
  t_start = cpu_second();
  execute_ga(selection_type, output_fname);
  fprintf(stdout,"[*] Finished in %f s\n", cpu_second() - t_start);

  free(output_fname);

  fprintf(stdout,"[*] Stopping device\n");
  reset_device();

  return EXIT_SUCCESS;
}
