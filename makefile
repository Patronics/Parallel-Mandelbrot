
fractal: fractal.c gfx.c
	gcc fractal.c gfx.c  -Wall -fopenmp -o fractal -lX11 -lm 

example: example.c gfx.c
	gcc example.c gfx.c -o example -lX11 -lm



clean:
	rm example fractal
