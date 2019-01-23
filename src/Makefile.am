.cu.o:
	$(NVCC) -o $@ -c $(NVCC_CFLAGS) --ptxas-options=-v --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" $<

CC = g++
CCLD=$(CC)
bin_PROGRAMS=cuRobby
cuRobby_SOURCES = main.c \
		  	prng.h prng.cu \
		 	test_device.h test_device.cu\
		  	utils.h utils.cu\
			world.h world.cu \
			ga.h ga.cu\
			robby.h robby.cu
cuRobby_CFLAGS  = $(CUDA_CFLAGS)
cuRobby_LDADD  = $(CUDA_LIBS) -lcurand
