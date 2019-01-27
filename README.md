# Introduction
This is the exact Genetic Algorithm used for solving the Robby at [robbyGA](https://github.com/minterciso/robbyga), but it'll execute on a CUDA device, instead of only on the host.

For a greater explanation of the Robby "problem", please refer to the host only code.

# CUDA differences
The CUDA implementation uses a CUDA enabled device to evolve all individuals in paralel on the device. For more information on how a CUDA device works, please refer to [nVidia](https://developer.nvidia.com/cuda-toolkit) website.

On this implementation each thread is responsible for executing exactly ONE individual, until all sessions are done. Each thread will create a new world per session per thread.

# Caveats found
I've found that it's way easier to keep all kernels on only one file and just use them from another file (with no kernels in it). This solved some issues with the cuRand module and simplified a lot
the code.

Also, since CUDA is good to work with warps (32 threads in each warp) increasing the amount of population from 200 to 256 did gave a little extra performance.

This CUDA code ran around 53% faster than the host only code, on a Geforce GTX 1060.
