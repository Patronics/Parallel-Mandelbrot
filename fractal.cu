/*
fractal.cu - Parallel interactive Mandelbrot Fractal Display
based on starting code for CSE 30341 Project 3.
*/
extern "C" {
#include "gfx.h"
}

#define WIDTH 1280
#define HEIGHT 960
//#define WIDTH 640
//#define HEIGHT 480

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include <cuda.h>

typedef struct coordSet {
	double xmin;
	double xmax;
	double ymin;
	double ymax;
	int maxiter;
	double xmid;
	double ymid;
	double xShift;
	double yShift;
	int zoom;
} coordSet;


struct colors {
	uint8_t r;
	uint8_t g;
	uint8_t b;
};

struct cache {
	struct colors hashmap[WIDTH * HEIGHT];
};

/*
Compute the number of iterations at point x, y
in the complex space, up to a maximum of maxiter.
Return the number of iterations at that point.

This example computes the Mandelbrot fractal:
z = z^2 + alpha

Where z is initially zero, and alpha is the location x + iy
in the complex plane.  Note that we are using the "complex"
numeric type in C, which has the special functions cabs()
and cpow() to compute the absolute values and powers of
complex values.
*/

__device__ int compute_point( double x, double y, int max )
{
	double z_real = 0;
	double z_imaginary = 0;
	double z_realsquared = 0;
	double z_imaginarysquared = 0;

	int iter = 0;
	for (iter = 0; iter < max; iter++) {
		z_imaginary = z_real * z_imaginary;
		z_imaginary = z_imaginary + z_imaginary + y;
		z_real = z_realsquared - z_imaginarysquared + x;
		z_realsquared = z_real * z_real;
		z_imaginarysquared = z_imaginary * z_imaginary;
		if (z_realsquared + z_imaginarysquared >= 4.0) {
			iter++;
			break;
		}
	}

	return iter;
}

/*
Compute an entire image, writing each point to the given bitmap.
Scale the image to the range (xmin-xmax,ymin-ymax).
*/

__global__ void compute_image(coordSet* coords, int width, int height, struct colors *colorsSet, struct cache* ch)
{
	double xmin=coords->xmin;
	double xmax=coords->xmax;
	double ymin=coords->ymin;
	double ymax=coords->ymax;
	double xShift=coords->xShift;
	double yShift=coords->yShift;
	int maxiter=coords->maxiter;
	int zoom=coords->zoom;

    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int my_j = blockDim.y * blockIdx.y + threadIdx.y;

	if (my_i < width && my_j < height) {
		double x = xmin + my_i*(xmax-xmin)/width;
		double y = ymin + my_j*(ymax-ymin)/height;
		int flipj;
		int j_test;
		int bounds = xmin - (xShift * width);

		if (yShift >= 0) 
			j_test = my_j - (yShift * height);
		else
			j_test = my_j + (yShift * height);
		
		flipj = (height -j_test) + 1;

		int key = ((my_i - (xShift * width)) + width * j_test);
		int key2 = ((my_i - (xShift * width)) + width * flipj);

		if (zoom != 0 || (ch->hashmap[key].r == 0 && ch->hashmap[key].g == 0 && ch->hashmap[key].b == 0) || xShift > 0 || (my_i <= bounds+2) || (my_j > abs(yShift * height) && yShift != 0)) {
			int iter = 0;
			iter = compute_point(x,y,maxiter);
			colorsSet[my_i+width*my_j].r = 255 * iter / maxiter;
			colorsSet[my_i+width*my_j].g = 255 * iter / (maxiter/30);
			colorsSet[my_i+width*my_j].b = 255 * iter / (maxiter/100);

			if (xShift == 0 && yShift == 0 && zoom == 0) {;
				ch->hashmap[key].r = colorsSet[my_i+width*my_j].r;
				ch->hashmap[key].g = colorsSet[my_i+width*my_j].g;
				ch->hashmap[key].b = colorsSet[my_i+width*my_j].b;
		
				ch->hashmap[key2].r = colorsSet[my_i+width*my_j].r;
       	              		ch->hashmap[key2].g = colorsSet[my_i+width*my_j].g;
               	      		ch->hashmap[key2].b = colorsSet[my_i+width*my_j].b;
			}
		}
		else {
			colorsSet[my_i+width*my_j].r = ch->hashmap[key].r;
			colorsSet[my_i+width*my_j].g = ch->hashmap[key].g;
			colorsSet[my_i+width*my_j].b = ch->hashmap[key].b;
		}
	}
}

void draw_point(int i, int j, struct colors c)
{
	gfx_color(c.r, c.g, c.b);
	// Plot the point on the screen.
	gfx_point(j, i);
}

void setMidpoints(coordSet* coords){
	coords->xmid = (coords->xmin+coords->xmax)/2;
	coords->ymid = (coords->ymin+coords->ymax)/2;
}

void reDraw(coordSet* coords){
	static struct cache* ch = (struct cache*)malloc(sizeof(struct cache));

    int width = gfx_xsize();
	int height = gfx_ysize();

    int n = width * height;
	
	#define BLOCK_SIZE 16 //TODO bigger blocks are likely faster
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	dim3 dimGrid(width/BLOCK_SIZE, height/BLOCK_SIZE); // 1*1 blocks in a grid

	struct colors* colorsSet;
	coordSet* cudaCoords;
	struct colors* c = (struct colors*)malloc(n * sizeof(struct colors));
	struct cache* cudaCache;

	cudaMalloc(&cudaCache, sizeof(struct cache));
	cudaMalloc(&colorsSet, n * sizeof(struct colors));
	cudaMalloc(&cudaCoords, sizeof(coordSet));

	// Show the configuration, just in case you want to recreate it.
	printf("coordinates: %lf %lf %lf %lf\n",coords->xmin,coords->xmax,coords->ymin,coords->ymax);
	// Display the fractal image

	struct timespec startTime, endTime;
	double runTime;
	clock_gettime(CLOCK_MONOTONIC, &startTime);
	// this is not the actual block size and thread count
	cudaError_t err = cudaMemcpy(cudaCoords, coords,sizeof(coordSet), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printf("%s memcpy0 coords\n", cudaGetErrorString(err));

	err = cudaMemcpy(colorsSet, c, n * sizeof(struct colors), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printf("%s memcpy1\n", cudaGetErrorString(err));

	cudaMemcpy(cudaCache, ch, sizeof(struct cache), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printf("%s memcpy2\n", cudaGetErrorString(err));

	compute_image <<<dimGrid, dimBlock>>>(cudaCoords, width, height, colorsSet, cudaCache);

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("%s synch\n", cudaGetErrorString(err));

	err = cudaMemcpy(c, colorsSet, n * sizeof(struct colors), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printf("%s memcpy3\n", cudaGetErrorString(err));

	err = cudaMemcpy(ch, cudaCache, sizeof(struct cache), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printf("%s memcpy4\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("%s synch2\n", cudaGetErrorString(err));
	
	clock_gettime(CLOCK_MONOTONIC, &endTime);
	runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	fprintf(stderr, "\ncalculating frame took %lf seconds\n", runTime);
	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			draw_point(i, j, c[i * width + j]);
		}
	clock_gettime(CLOCK_MONOTONIC, &endTime);
	runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	fprintf(stderr, "\ncalculating and rendering frame took %lf seconds\n", runTime);

	free(c);
	cudaFree(colorsSet);
	cudaFree(cudaCoords);
	cudaFree(cudaCache);
}


void zoomIn(coordSet* coords,double extent){
	setMidpoints(coords);
	double width = coords->xmid-coords->xmin;
	double height = coords->ymid-coords->ymin;
	coords->xmax=coords->xmid+(width/extent);
	coords->xmin=coords->xmid-(width/extent);
	coords->ymax=coords->ymid+(height/extent);
	coords->ymin=coords->ymid-(height/extent);
	coords->zoom -= 1;
	setMidpoints(coords);
	reDraw(coords);
}

void zoomOut(coordSet* coords, double extent){
	setMidpoints(coords);
	double width = coords->xmid-coords->xmin;
	double height = coords->ymid-coords->ymin;
	coords->xmax=coords->xmid+(width*extent);
	coords->xmin=coords->xmid-(width*extent);
	coords->ymax=coords->ymid+(height*extent);
	coords->ymin=coords->ymid-(height*extent);
	coords->zoom += 1;
	setMidpoints(coords);
	reDraw(coords);
}

void shiftFrame(coordSet* coords, double xShift, double yShift){
	setMidpoints(coords);
	double width = coords->xmax-coords->xmin;
	double height = coords->ymax-coords->ymin;
	coords->xmax+=xShift*width;
	coords->xmin+=xShift*width;
	coords->ymax+=yShift*height;
	coords->ymin+=yShift*height;
	coords->xShift+=xShift;
	coords->yShift+=yShift;
	setMidpoints(coords);
	reDraw(coords);
}


//accidentally discovered, mirrors coords, may be useful
void reflect(coordSet* coords){
	setMidpoints(coords);
	coords->xmax=coords->xmax+(coords->xmid-coords->xmax)*2;
	coords->xmin=coords->xmin+(coords->xmid-coords->xmin)*2;
	coords->ymax=coords->ymax+(coords->ymid-coords->ymax)*2;
	coords->ymin=coords->ymin+(coords->ymid-coords->ymin)*2;
	setMidpoints(coords);
	reDraw(coords);
}


int main( int argc, char *argv[] ){
	// The initial boundaries of the fractal image in x,y space.
	const double xminDefault = -1.5;
	const double xmaxDefault = 0.5;
	const double yminDefault = -1.0;
	const double ymaxDefault=  1.0;
	// Maximum number of iterations to compute.
	// Higher values take longer but have more detail.
	const int maxiterDefault = 3000; //default 500
	
	coordSet* dispCoords = (coordSet*)malloc(sizeof(coordSet));
	
	if(argv[1] && argv[2] && argv[3] && argv[4] && argv[5]){
		dispCoords->xmin = atof(argv[1]);
		dispCoords->xmax = atof(argv[2]);
		dispCoords->ymin = atof(argv[3]);
		dispCoords->ymax = atof(argv[4]);
		dispCoords->maxiter = atoi(argv[5]);
		dispCoords->xShift = 0;
		dispCoords->yShift = 0;
		dispCoords->zoom = 0;
		setMidpoints(dispCoords);
	}else{
		dispCoords->xmin=xminDefault;
		dispCoords->xmax=xmaxDefault;
		dispCoords->ymin=yminDefault;
		dispCoords->ymax=ymaxDefault;
		dispCoords->maxiter=maxiterDefault;
		dispCoords->xShift = 0;
		dispCoords->yShift = 0;
		dispCoords->zoom = 0;
		setMidpoints(dispCoords);
	}


	// Open a new window.
	gfx_open(WIDTH,HEIGHT,"Mandelbrot Fractal");


	// Fill it with a dark blue initially.
	gfx_clear_color(0,0,255);
	gfx_clear();

	//draw intial position
	reDraw(dispCoords);


	while(1) {
		// Wait for a key or mouse click.
		int c = gfx_wait();
		printf("got character %c\n",c);
		// Quit if q is pressed.
		switch(c){
		case 'q':
			free(dispCoords);
			exit(0);
		//reset default position
		case 'r':
			dispCoords->xmin=xminDefault;
			dispCoords->xmax=xmaxDefault;
			dispCoords->ymin=yminDefault;
			dispCoords->ymax=ymaxDefault;
			setMidpoints(dispCoords);
			reDraw(dispCoords);
			break;
		//Reflect the view (mirroring it)
		case 'R':
			reflect(dispCoords);
			break;
		//zoom in/out with i/o (or smoothly with I/O)
		case 'i':
			printf("zooming in\n");
			zoomIn(dispCoords, 2);
			break;
		case 'o':
			printf("zooming out\n");
			zoomOut(dispCoords, 2);
			break;
		case 'I':
			printf("zooming in slightly\n");
			zoomIn(dispCoords, 1.25);
			break;
		case 'O':
			printf("zooming out slightly\n");
			zoomOut(dispCoords, 1.25);
			break;
		//pan with wasd (or smoothly with WASD)
		case 'w':
			shiftFrame(dispCoords, 0, -0.5);
			break;
		case 'W':
			shiftFrame(dispCoords, 0, -0.25);
			break;
		case 's':
			shiftFrame(dispCoords, 0, 0.5);
			break;
		case 'S':
			shiftFrame(dispCoords, 0, 0.25);
			break;
		case 'a':
			shiftFrame(dispCoords, -0.5, 0);
			break;
		case 'A':
			shiftFrame(dispCoords, -0.25, 0);
			break;
		case 'd':
			shiftFrame(dispCoords, 0.5, 0);
			break;
		case 'D':
			shiftFrame(dispCoords, 0.25, 0);
			break;
		}
//		} else if(c=='q'){
	}

	return 0;
}
