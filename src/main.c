
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "test_device.h"

int main(int argc, char **argv)
{
    fprintf(stdout,"[*] Starting device\n");
    start_device();

    fprintf(stdout,"[*] Testing the device\n");
    test_prng();
    test_uniform_prng(0,100);
    test_world_creation(100);
    fprintf(stdout,"[*] Stopping device\n");
    reset_device();

    return EXIT_SUCCESS;
}
