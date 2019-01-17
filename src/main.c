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
    fprintf(stdout,"[*] Stopping device\n");
    reset_device();

    return EXIT_SUCCESS;
}
