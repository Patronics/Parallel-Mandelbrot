
all: fractal serialfractal

fractal: fractal.c gfx.c
	gcc fractal.c gfx.c -O3 -Wall -fopenmp -o fractal -lX11 -lm

serialfractal: fractal.c gfx.c
	gcc fractal.c gfx.c -O3 -Wall -Wno-unknown-pragmas -o serialfractal -lX11 -lm



example: example.c gfx.c
	gcc example.c gfx.c -o example -lX11 -lm



clean:
	rm -f example fractal serialfractal
