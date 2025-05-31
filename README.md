# CUDA Accelerated Barnes-Hut N-Body Simulation

#### 512k bodies running in real-time on a NVIDIA RTX 500 Ada Generation Laptop GPU.

![](red-clusters.gif)

#### 1mln bodies

![](1mln-clusters.png)

## TODO

- fit octree to bbox
- power spectrum ic
- janus model
- compute energy
- compare acc error with all-pairs
- revert to 32 sized groups when traversing (apparently faster)
- quadruple moments
- switch to trailing underscore for private members, remove leading underscore for private methods
