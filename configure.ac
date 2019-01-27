# ------------------------------------------------------------------------------
AC_PREREQ([2.64])
AC_INIT([cuRobby], [0.1], [minterciso@gmail.com])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_HEADERS([config.h])

ARCH=`uname -m | grep -o 64`
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   [  --with-cuda=PATH  		    prefix where cuda is installed [default=auto]])
if test -n "$with_cuda"
then
   CUDA_CFLAGS="-I$with_cuda/include"
   CUDA_LIBS="-L$with_cuda/lib$ARCH -lcudart"
   NVCC="$with_cuda/bin/nvcc"
   CUDA_PATH="$with_cuda"
else
   CUDA_CFLAGS="-I/usr/local/cuda/include"
   CUDA_LIBS="-L/usr/local/cuda/lib$ARCH -lcudart"
   NVCC="nvcc"
   CUDA_PATH="/usr/local/cuda/"
fi
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(CUDA_PATH)

DEBUG=false
AC_ARG_ENABLE([debug],
        AS_HELP_STRING([--enable-debug], [Turn on device debug]))
AS_IF([test "x$enable_debug" == "xyes"], [
        DEBUG=true
        AC_DEFINE([DEBUG], [1], [Debug current program])
        ])

# ------------------------------------------------------------------------------
# Setup nvcc flags
# ------------------------------------------------------------------------------
if test x$DEBUG == xtrue 
then
   AC_MSG_NOTICE([Enabling DEBUG])
   NVCCFLAGS="-g -G"
   NVCC_CFLAGS="-g -G"
   CFLAGS="-g -ggdb"
else
   AC_MSG_NOTICE([Enabling Performance])
   NVCCFLAGS="-O3 -use_fast_math"
   NVCC_CFLAGS="-O3 -use_fast_math"
   CFLAGS="-O3 -fexpensive-optimizations -use_fast_math"
fi
AC_SUBST(NVCCFLAGS)
AC_SUBST(NVCC_CFLAGS)


# Checks for programs.
AC_PROG_CC
AM_PROG_CC_C_O

# Checks for libraries.
AC_CHECK_LIB([m], [pow])

# Checks for header files.
AC_CHECK_HEADERS([limits.h stdlib.h string.h math.h $CUDA_PATH/include/cuda.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])


AC_CONFIG_FILES([Makefile
                 src/Makefile])

AC_OUTPUT
