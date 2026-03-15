CC := gcc
AR := ar
STD := c17
CFLAGS := -Wall -O3

#LINKING OPENCL:
#sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so

INCLUDE_DIR := -I./src/mlpc -I./src/ddpgc -I/opt/rocm-6.4.4/include

MLPC_SRCS := \
	mlp.c \
	mlp_multi.c \
	matrix.c \
	matrix_cl.c \
	activation.c \
	loss.c \
	adam.c \
	random.c

DDPGC_SRCS := \
	ddpg.c

MLPC_OBJS := $(MLPC_SRCS:%.c=./build/mlpc/%.o)
DDPGC_OBJS := $(DDPGC_SRCS:%.c=./build/ddpgc/%.o)

.PHONY: all clean

all: ./lib/mlpc.a ./lib/ddpgc.a ./bin/saddle ./bin/pendulum ./bin/target_seeker ./bin/target_seeker_mlp ./bin/open_cl_test ./bin/boost_converter ./bin/shooter

./lib/mlpc.a: $(MLPC_OBJS)
	@echo "Linking $@"
	@mkdir -p $(dir $@)
	@$(AR) rcs $@ $(MLPC_OBJS)

./build/mlpc/%.o: ./src/mlpc/%.c
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $(INCLUDE_DIR) -c $< -o $@

./lib/ddpgc.a: $(DDPGC_OBJS)
	@echo "Linking $@"
	@mkdir -p $(dir $@)
	@$(AR) rcs $@ $(DDPGC_OBJS)

./build/ddpgc/%.o: ./src/ddpgc/%.c
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $(INCLUDE_DIR) -c $< -o $@ 

./bin/saddle: ./examples/saddle.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/mlpc.a -lm -lOpenCL -o $@

./bin/pendulum: ./examples/pendulum.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/ddpgc.a ./lib/mlpc.a -lm -lOpenCL -o $@

./bin/target_seeker: ./examples/target_seeker.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/ddpgc.a ./lib/mlpc.a -lm -lOpenCL -o $@

./bin/target_seeker_mlp: ./examples/target_seeker_mlp.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/mlpc.a -lm -lOpenCL -o $@

./bin/open_cl_test: ./examples/open_cl_test.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/mlpc.a -lm -lOpenCL -o $@

./bin/boost_converter: ./examples/boost_converter.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/ddpgc.a ./lib/mlpc.a -lm -lOpenCL -o $@

./bin/shooter: ./examples/shooter.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< $(INCLUDE_DIR) ./lib/ddpgc.a ./lib/mlpc.a -lm -lOpenCL -o $@

clean:
	@rm -rf ./build
	@rm -rf ./lib
	@rm -rf ./bin
