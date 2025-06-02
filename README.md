# CUDA Accelerated Barnes-Hut N-Body Simulation

#### 1mln bodies

![](1mln-clusters.png)

#### 512k bodies running in real-time on a NVIDIA RTX 500 Ada Generation Laptop GPU.

![](red-clusters.gif)

## TODO

- fit octree to bbox
- power spectrum ic
- janus model
- revert to 32 sized groups when traversing (apparently faster)
- quadruple moments
- switch to trailing underscore for private members, remove leading underscore for private methods
- refactor initial configuration
