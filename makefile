example: example.c gfx.c
	gcc example.c gfx.c -o example -lX11 -lm



clean:
	rm example
