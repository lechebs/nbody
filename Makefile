SRC_DIR = src
BUILD_DIR = build

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20

SDL2_CFLAGS := $(shell pkg-config --cflags sdl2)
SDL2_LIBS   := $(shell pkg-config --libs sdl2)
GLEW_CFLAGS := $(shell pkg-config --cflags glew)
GLEW_LIBS   := $(shell pkg-config --libs glew)
OPENGL_LIBS := -lGL

CXXFLAGS += $(SDL2_CFLAGS) $(GLEW_CFLAGS)
LDLIBS   := $(SDL2_LIBS) $(GLEW_LIBS) $(OPENGL_LIBS)

INCLUDE = -Iinclude

NVCC = nvcc
CUDA_ARCHS ?= 50 60 70 75 80 89

NVCC_GENCODE := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NVCC_PTX := -gencode arch=compute_$(lastword $(CUDA_ARCHS)),code=compute_$(lastword $(CUDA_ARCHS))
NVCCFLAGS := $(NVCC_GENCODE) $(NVCC_PTX) -maxrregcount=40

EXE = main
CUDA_OBJS = points.o btree.o octree.o physics_common.o barnes_hut.o simulation.o validator.o
OBJS = main.o renderer.o shader_program.o camera.o $(CUDA_OBJS)

all: $(BUILD_DIR) $(EXE)

$(EXE): $(addprefix $(BUILD_DIR)/, $(OBJS))
	$(NVCC) $^ $(NVCCFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDE) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/cuda/%.cu
	$(NVCC) -dc $(NVCCFLAGS) $(INCLUDE) $^ -o $@

.PHONY: clean

clean:
	rm -r $(BUILD_DIR)
