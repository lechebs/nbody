# CUDA Accelerated Barnes-Hut N-Body Simulation

#### 1mln bodies

![](1mln-clusters.png)

#### Octree visualization

![](spinning.png)
![](spinning-octree.png)

#### 256k bodies

![](disk.png)

#### 256k bodies per disk

![](2disks.gif)

## TODO

- fit octree to bbox
- revert to 32 sized groups when traversing (apparently faster)
- quadruple moments
- switch to trailing underscore for private members, remove leading underscore for private methods
- traversal queue allocation
- plummer for accuracy

## Future work

- replace cub and thrust primitives with custom kernels
- power spectrum ic
- janus cosmological model


