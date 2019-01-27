.cu.o:
	$(NVCC) -o $@ -c $(NVCC_CFLAGS) --ptxas-options=-v --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" $<

CC = g++
CCLD=$(CC)
bin_PROGRAMS=cuRobby
cuRobby_SOURCES = main.c \
		  	utils.h utils.cu\
			ga.h ga.cu\
			kernels.h kernels.cu \
			read_parameters.h read_parameters.c
cuRobby_CFLAGS  = $(CUDA_CFLAGS)
cuRobby_LDADD  = $(CUDA_LIBS) -lcurand
