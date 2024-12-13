SRC_DIR = src
BUILD_DIR = build

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20
LDLIBS = -lGL -lGLEW -lSDL2
INCLUDE = -Iinclude

EXE = main
OBJS = main.o Renderer.o ShaderProgram.o Camera.o

all: $(BUILD_DIR) $(EXE)

$(EXE): $(addprefix $(BUILD_DIR)/, $(OBJS))
	$(CXX) $^ $(LDLIBS) -o $@ 

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDE) $^ -o $@

.PHONY: clean

clean:
	rm -r $(BUILD_DIR)
