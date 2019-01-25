
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "utils.h"
#include "test_device.h"
#include "ga.h"

const char *prog_name;

void usage(FILE *stream, int exit_code){
  fprintf(stream, "Usage: %s options\n", prog_name);
  fprintf(stream, "\t-h\t\t--help\t\tThis help message\n");
  fprintf(stream, "\t-t\t\t--test\t\tMake some very basic testing on the device.");
  exit(exit_code);
}

int main(int argc, char **argv)
{
  // Program options
  prog_name = argv[0];
  const char* const short_options = "ht";
  const struct option long_options[] = {
      {"help", 0, NULL, 'h'},
      {"test", 0, NULL, 't'},
      {0,0,0,0}
  };
  int next_option;
  int test = 0;
  do{
    next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    switch(next_option){
      case 'h': usage(stdout,EXIT_SUCCESS); break;
      case 't': test=1; break;
      case '?': usage(stderr,EXIT_FAILURE); break;
      case -1: break;
      default: abort();
    }
  }while(next_option != -1);

  fprintf(stdout,"[*] Starting device\n");
  start_device();

  if(test == 1){
    fprintf(stdout,"[*] Testing the device\n");
    test_prng();
    test_uniform_prng(0,100);
    test_world_creation(100);
    test_population_creation(200);
  }
  else{
    execute_ga();
  }
  fprintf(stdout,"[*] Stopping device\n");
  reset_device();

  return EXIT_SUCCESS;
}
