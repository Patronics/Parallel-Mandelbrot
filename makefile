
all: fractal serialfractal

fractal: fractal_parallel.c gfx.c
	gcc -O3 fractal_parallel.c gfx.c  -Wall -fopenmp -o fractal -lX11 -lm 

serialfractal: fractal_serial.c gfx.c
	gcc fractal_serial.c gfx.c  -Wall -Wno-unknown-pragmas -o serialfractal -lX11 -lm 



example: example.c gfx.c
	gcc example.c gfx.c -o example -lX11 -lm



clean:
	rm -f example fractal serialfractal
