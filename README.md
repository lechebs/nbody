# Real-Time CUDA Accelerated Barnes-Hut N-Body Simulation

A tree code implementation for the Barnes-Hut algorithm, running in real-time on the GPU, developed using CUDA C++ and OpenGL, which scales to systems of few million particles on a NVIDIA RTX 500 Ada Laptop GPU.

![](screenshots/1mln-clusters.png)

## Prerequisites

The program was developed and tested on Ubuntu 22.04 LTS using the CUDA Toolkit 12.9, SDL2 and OpenGL 4.3 core.
The basic dependencies can be installed under Ubuntu by running
```
sudo apt install build-essential libsdl2-dev libglew-dev
```

Detailed installation instructions for the CUDA Toolkit can be found at [here](https://developer.nvidia.com/cuda-downloads).

## Building

To build the project for a GPU with compute capability `xy`, run
```
make CUDA_ARCHS="xy"
```
More info on compute capability versioning [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list). If not specified, the program will be built for the following architectures: `50 60 70 75 80 89`.

Build files will be placed under `build/`, running
```
make clean
```
will remove the `build/` directory but keep the executable.

## Usage

The program can be executed by running
```
./main
```

Drag to orbit the camera around the origin, and scroll to zoom. Pressing `o` key toggles the octree visualization, `space` pauses/resumes the simulation.

## Gallery

| ![](screenshots/spinning.png) |  ![](screenshots/spinning-octree.png) |
|:--------:|:-------:|
| Basic visualization | Octree visualization |

| ![](screenshots/disk.png) |  ![](screenshots/shell-galaxy.png) |
|:--------:|:-------:|
| Self-gravitating disk of 262,144 particles | Cold collapse of uniform sphere |

| ![](screenshots/disk.gif) |  ![](screenshots/2disks.gif) |
|:--------:|:-------:|
| Disk of 524,288 particles running at ~37 FPS (θ=0.75) | Two disks of 262,144 particles each running at ~30 FPS (θ=0.75) |

### 2,097,152 particles at ~150ms per step (θ=0.6)

![](screenshots/2mln-explosion.gif)

An additional video can be found [here](https://drive.google.com/file/d/1YTa5hYdYPj_kloaZWec7PzCBgfg7Z7SG/view?usp=sharing).

## Bibliography

- Tero Karras. 2012. Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. In Proceedings of the Fourth ACM SIGGRAPH / Eurographics conference on High-Performance Graphics (EGGH-HPG'12). Eurographics Association, Goslar, DEU, 33–37.

- Robin Cazalbou, Florent Duchaine, Eric Quémerais, Bastien Andrieu, Gabriel Staffelbach, and Bruno Maugars. 2024. Hybrid Multi-GPU Distributed Octrees Construction for Massively Parallel Code Coupling Applications. In Proceedings of the Platform for Advanced Scientific Computing Conference (PASC '24). Association for Computing Machinery, New York, NY, USA, Article 14, 1–11. https://doi.org/10.1145/3659914.3659928

- Jeroen Bédorf, Evghenii Gaburov, and Simon Portegies Zwart. 2012. A sparse octree gravitational N-body code that runs entirely on the GPU processor. Journal of Computational Physics 231, 7 (2012), 2825–2839. DOI:https://doi.org/https://doi.org/10.1016/j.jcp.2011.12.024



