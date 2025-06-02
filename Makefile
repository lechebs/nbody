SRC_DIR = src
BUILD_DIR = build

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20
LDLIBS = -lGL -lGLEW -lSDL2
INCLUDE = -Iinclude

NVCC = nvcc
NVCCFLAGS = -arch=compute_89 -code=sm_89 -maxrregcount=40

EXE = main
CUDA_OBJS = btree.o octree.o barnes_hut.o wrappers.o validator.o
OBJS = main.o Renderer.o ShaderProgram.o Camera.o $(CUDA_OBJS)

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
