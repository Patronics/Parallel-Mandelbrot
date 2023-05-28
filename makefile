
all: fractal serialfractal cudafractal
 

ifeq ($(OS),Windows_NT) 
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
endif

CC = gcc

ifeq ($(detected_OS),Darwin)    #MacOS
	#MacOS X11 Libs need linking in separately
	CFLAGS += -I/usr/X11R6/include/ -L/usr/X11R6/lib/
	CC = gcc-13
	CU = nvcc
endif

fractal: fractal.c gfx.c
	$(CC) fractal.c gfx.c -Wall $(CFLAGS) -fopenmp -o fractal -lX11 -lm

serialfractal: fractal.c gfx.c
	$(CC) fractal.c gfx.c  -Wall $(CFLAGS) -Wno-unknown-pragmas -o serialfractal -lX11 -lm

cudafractal: fractal.cu gfx.c
	$(CC) fractal.cu gfx.c $(CFLAGS) -o cudafractal-lx11 -lm

example: example.c gfx.c
	gcc example.c gfx.c -o example -lX11 -lm

clean:
	rm -f example fractal serialfractal
