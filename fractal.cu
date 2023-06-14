/*
fractal.cu - Parallel interactive Mandelbrot Fractal Display
based on starting code for CSE 30341 Project 3.
*/
extern "C" {
#include "gfx.h"
}

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>

#include <cuda.h>

int blockSize;
int blockCount;
int windowWidth;
int windowHeight;

typedef struct coordSet {
	double xmin;
	double xmax;
	double ymin;
	double ymax;
	int maxiter;
	double xmid;
	double ymid;
} coordSet;


struct colors {
	int r;
	int g;
	int b;
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

__global__ void compute_image(coordSet* coords, int width, int height, struct colors *colorsSet, int blockCount, int blockSize, double balance)
{
	double xmin=coords->xmin;
	double xmax=coords->xmax;
	double ymin=coords->ymin;
	double ymax=coords->ymax;
	int maxiter=coords->maxiter;
	int my_a = (blockDim.x * blockIdx.x + threadIdx.x) % width;
	int my_b = (blockDim.x * blockIdx.x + threadIdx.x) / width;
	int stepx = (blockCount * blockSize) % width;
	int stepy = (blockCount * blockSize) / width;
	int carry;

    //int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    //int my_j = blockDim.y * blockIdx.y + threadIdx.y;

	for(int my_i = my_a, my_j = my_b; my_i < width && my_j < height*balance; my_i = (my_i + stepx) % width, my_j = my_j + stepy + carry) {
		carry = 0;
    	double x = xmin + my_i*(xmax-xmin)/width;
		double y = ymin + my_j*(ymax-ymin)/height;

    	int iter = compute_point(x,y,maxiter);
    	colorsSet[my_i+width*my_j].r = 255 * iter / maxiter;
		colorsSet[my_i+width*my_j].g = 255 * iter / (maxiter/30);
		colorsSet[my_i+width*my_j].b = 255 * iter / (maxiter/100);

		if(my_i + stepx >= width) carry = 1;
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
    int width = gfx_xsize();
	int height = gfx_ysize();

    int n = width * height;
	double balance = 1.0;
	if(n > 100000000) balance = 0.6;
	
	//#define BLOCK_SIZE 16 //TODO bigger blocks are likely faster
	
	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	//dim3 dimGrid(width/BLOCK_SIZE, height/BLOCK_SIZE); // 1*1 blocks in a grid
	dim3 dimGrid(width*width/blockCount, height*height/blockCount);

	struct colors* colorsSet;
	struct colors* c = (struct colors*)malloc(n * sizeof(struct colors));
	cudaMalloc(&colorsSet, n * sizeof(struct colors));

	coordSet* cudaCoords;
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

	if (balance > 0.0) {
		compute_image <<<blockCount, blockSize>>>(cudaCoords, width, height, colorsSet, blockCount, blockSize, balance);

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) printf("%s synch0\n", cudaGetErrorString(err));

		err = cudaMemcpy(c, colorsSet, n * sizeof(struct colors), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) printf("%s memcpy2\n", cudaGetErrorString(err));

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) printf("%s synch1\n", cudaGetErrorString(err));
	}
	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			draw_point(i, j, c[i * width + j]);
		}

	clock_gettime(CLOCK_MONOTONIC, &endTime);
	runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	
	printf("Blocks: %d\tThreads per Block: %d\tSize:%dx%d\tDepth: %d\tTime: %f\n",
	blockSize, blockCount, width, height, coords->maxiter, runTime);

	free(c);
	cudaFree(colorsSet);
	cudaFree(cudaCoords);
}


void zoomIn(coordSet* coords,double extent){
	setMidpoints(coords);
	double width = coords->xmid-coords->xmin;
	double height = coords->ymid-coords->ymin;
	coords->xmax=coords->xmid+(width/extent);
	coords->xmin=coords->xmid-(width/extent);
	coords->ymax=coords->ymid+(height/extent);
	coords->ymin=coords->ymid-(height/extent);
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

void usage(){
    printf("Usage: benchmark [n] [m] [dim] [max_iter]\n");
    printf("\tn\t\t=\tnumber of blocks (defaults to 512)\n");
    printf("\tm\t\t=\tthreads per block (defaults to 512)\n");
    printf("\tdim\t\t=\twidth/height of canvas in pixels (defaults to 1600)\n");
    printf("\tmax_iter\t=\tmax iterations (defaults to 100)\n\n");
    exit(1);
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
	windowWidth     = 640;
	windowHeight    = 480;
	blockCount 		= 256;
	blockSize	    = 256;
	
	coordSet* dispCoords = (coordSet*)malloc(sizeof(coordSet));
	
	if(argc>5){
		dispCoords->xmin = atof(argv[1]);
		dispCoords->xmax = atof(argv[2]);
		dispCoords->ymin = atof(argv[3]);
		dispCoords->ymax = atof(argv[4]);
		dispCoords->maxiter = atoi(argv[5]);
		setMidpoints(dispCoords);
	}else{
		dispCoords->xmin=xminDefault;
		dispCoords->xmax=xmaxDefault;
		dispCoords->ymin=yminDefault;
		dispCoords->ymax=ymaxDefault;
		dispCoords->maxiter=maxiterDefault;
		setMidpoints(dispCoords);
	}

	if(argc>9)
	{
		dispCoords->maxiter = atoi(argv[5]);
		windowWidth = atoi(argv[6]);
		windowHeight = atoi(argv[7]);
		blockCount = atoi(argv[8]);
		blockSize = atoi(argv[9]);
	}

	// Open a new window.
	gfx_open(windowWidth,windowHeight,"Mandelbrot Fractal");


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
