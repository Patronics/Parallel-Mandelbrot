all: benchmark fractal fractal-nox serialfractal serialfractal-nox cudafractal_cache cudafractal_cache-nox cudafractal_loadbalance cudafractal_loopbreak
 

ifeq ($(OS),Windows_NT) 
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
endif

CC = gcc

## Debug flag, run 'make D=1' to enable
ifneq ($(D),1)
	CFLAGS	+= -O3
	NVCCFLAGS += -O3
else
	CFLAGS	+= -g3 -Og
	#nvcc doesn't support most of the optimization and debugging flags
	NVCCFLAGS += -g -G
endif

ifeq ($(detected_OS),Darwin)    #MacOS
	#MacOS X11 Libs need linking in separately
	CFLAGS += -I/usr/X11R6/include/ -L/usr/X11R6/lib/
	CC = gcc-13
endif

benchmark: benchmark.c
	$(CC) benchmark.c -Wall $(CFLAGS) -o benchmark -lm
	ln -s benchmark Xbenchmark

fractal: fractal.c gfx.c
	$(CC) fractal.c gfx.c -Wall $(CFLAGS) -fopenmp -o fractal -lX11 -lm

fractal-nox: fractal.c
	$(CC) fractal.c -Wall $(CFLAGS) -Wno-unused-variable -fopenmp -o fractal-nox -lm -D NOX -D OPENMP

serialfractal: fractal.c gfx.c
	$(CC) fractal.c gfx.c  -Wall $(CFLAGS) -Wno-unknown-pragmas -o serialfractal -lX11 -lm

serialfractal-nox: fractal.c
	$(CC) fractal.c -Wall $(CFLAGS) -Wno-unused-variable -Wno-unknown-pragmas -o serialfractal-nox -lm -D NOX

cudafractal_cache: fractal_cache.cu gfx.c
	nvcc fractal_cache.cu gfx.c -Xcompiler -fopenmp $(NVCCFLAGS) -o cudafractal_cache -lX11 -lm

cudafractal_cache-nox: fractal_cache.cu
	nvcc fractal_cache.cu -Xcompiler -fopenmp $(NVCCFLAGS) -o cudafractal_cache-nox -lm -D NOX

cudafractal_loadbalance: fractal_cache.cu gfx.c
	nvcc fractal_loadbalance.cu gfx.c -Xcompiler -fopenmp $(NVCCFLAGS) -o cudafractal_loadbalance -lX11 -lm

cudafractal_loopbreak: fractal_cache.cu gfx.c
	nvcc fractal_loopbreak.cu gfx.c -Xcompiler -fopenmp $(NVCCFLAGS) -o cudafractal_loopbreak -lX11 -lm

example: example.c gfx.c
	gcc example.c gfx.c -o example -lX11 -lm

clean:
	rm -f example fractal serialfractal cudafractal_cache cudafractal_loadbalance benchmark cudafractal_loopbreak
