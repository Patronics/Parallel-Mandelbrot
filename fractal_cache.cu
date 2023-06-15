/*
fractal.cu - Parallel interactive Mandelbrot Fractal Display
based on starting code for CSE 30341 Project 3.
*/
#ifndef NOX
extern "C" {
#include "gfx.h"
}
#endif

#define BENCHMARK

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include <cuda.h>
#include <omp.h>

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


int blockSize;
int blockCount;
int windowWidth;
int windowHeight;
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

__global__ void compute_image(coordSet* coords, int width, int height, struct colors *colorsSet, struct colors* ch, int blockCount, int blockSize, double balance)
{
	double xmin=coords->xmin;
	double xmax=coords->xmax;
	double ymin=coords->ymin;
	double ymax=coords->ymax;
	double xShift=coords->xShift;
	double yShift=coords->yShift;
	int maxiter=coords->maxiter;
	int zoom=coords->zoom;
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
		int flip_j;
                int i_adj, j_adj;
		
                bool l_x = (xShift <= -0.75) || (xShift < 0 && my_i <= (1 - abs(xShift)) * width);
		bool m_x = (xShift >= 0.75)  || (xShift > 0 && my_i >= (xShift * width));
		bool l_y = (yShift <= -0.75) || (yShift < 0 && my_j <= (1 - abs(yShift)) * height);
		bool m_y = (yShift >= 0.75)  || (yShift > 0 && my_j >= (yShift * height));

		i_adj = my_i + (xShift * width);
		j_adj = my_j + (yShift * height);

		flip_j = (height - j_adj) + 1;

		int key =  (i_adj + width * j_adj);
		int key2 = (i_adj + width * flip_j);

		if (zoom != 0 || ch[key].r == 0 || l_x || m_x || l_y || m_y) {
			int iter = 0;
			iter = compute_point(x,y,maxiter);
			colorsSet[my_i+width*my_j].r = 255 * iter / maxiter;
			colorsSet[my_i+width*my_j].g = 255 * iter / (maxiter/30);
			colorsSet[my_i+width*my_j].b = 255 * iter / (maxiter/100);

			if (xShift == 0 && yShift == 0 && zoom == 0) {;
				ch[key].r = colorsSet[my_i+width*my_j].r;
				ch[key].g = colorsSet[my_i+width*my_j].g;
				ch[key].b = colorsSet[my_i+width*my_j].b;
		
				ch[key2].r = colorsSet[my_i+width*my_j].r;
       	              		ch[key2].g = colorsSet[my_i+width*my_j].g;
               	      		ch[key2].b = colorsSet[my_i+width*my_j].b;
			}
		}
		else {
			colorsSet[my_i+width*my_j].r = ch[key].r;
			colorsSet[my_i+width*my_j].g = ch[key].g;
			colorsSet[my_i+width*my_j].b = ch[key].b;
		}
		if(my_i + stepx >= width) carry = 1;
	
		
	}
}
uint16_t compute_pointCPU( double x, double y, uint16_t max )
{
        double z_real = 0;
        double z_imaginary = 0;
        double z_realsquared = 0;
        double z_imaginarysquared = 0;

        uint16_t iter = 0;
	//if(x > -0.6 && x < 0.2 && y < -0.3 && y > 0.3) return max;

       	for (iter = 0; iter < max; ++iter) {
                z_imaginary = z_real * z_imaginary;
                z_imaginary = z_imaginary + z_imaginary + y;
                z_real = z_realsquared - z_imaginarysquared + x;
                z_realsquared = z_real * z_real;
                z_imaginarysquared = z_imaginary * z_imaginary;
                if (z_realsquared + z_imaginarysquared >= 4.0) {
                        ++iter;
                        break;
                }
        }

        return iter;
}

void compute_imageCPU(coordSet* coords, int width, int height, struct colors *colorsSet, double balance)
{
        double xmin=coords->xmin;
        double xmax=coords->xmax;
        double ymin=coords->ymin;
        double ymax=coords->ymax;
        int maxiter=coords->maxiter;
	double start_time = omp_get_wtime();
    //int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    //int my_j = blockDim.y * blockIdx.y + threadIdx.y;
        //int total_threads = gridDim.x * blockDim.x;
        //int total_threads1 = gridDim.y * blockDim.y;
	#pragma omp parallel for schedule(dynamic)
        for(int i = height*balance; i < height; ++i) {

		#pragma omp parallel for schedule(dynamic)
		for(int j = 0; j < width; ++j) {

    double x = xmin + j*(xmax-xmin)/width;
        double y = ymin + i*(ymax-ymin)/height;

    uint16_t iter = compute_pointCPU(x,y,maxiter);
    colorsSet[i*width+j].r = 255 * iter / maxiter;
        colorsSet[i*width+j].g = 255 * iter / (maxiter/30);
        colorsSet[i*width+j].b = 255 * iter / (maxiter/100);
	//colorsSet[i*width+j].r = (colorsArray[iter] & 0xFF0000) >> 16;
        //colorsSet[i*width+j].b = (colorsArray[iter] & 0xFF00) >> 8;
        //colorsSet[i*width+j].g = colorsArray[iter] & 0xFF;
        }
	}
	double end_time = omp_get_wtime();
        printf("%.5f\n", end_time - start_time);
}

void draw_point(int i, int j, struct colors c)
{
	#ifndef NOX
	gfx_color(c.r, c.g, c.b);
	// Plot the point on the screen.
	gfx_point(j, i);
	#endif
}

void setMidpoints(coordSet* coords){
	coords->xmid = (coords->xmin+coords->xmax)/2;
	coords->ymid = (coords->ymin+coords->ymax)/2;
}

void reDraw(coordSet* coords){
	
    int width = windowWidth;
	int height = windowHeight;

    int n = width * height;
	
	static struct colors* ch = (struct colors*)malloc(n*sizeof(struct colors));


    double balance = 1.0; //always use GPU based rendering on this version
	
	//#define BLOCK_SIZE 16 //TODO bigger blocks are likely faster
	
	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	dim3 dimGrid(width*width/blockCount, height*height/blockCount); // 1*1 blocks in a grid

	struct colors* colorsSet;
	coordSet* cudaCoords;
	struct colors* c = (struct colors*)malloc(n * sizeof(struct colors));
	struct colors* cudaCache;

	cudaMalloc(&cudaCache, n * sizeof(struct colors));
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

	//cudaMemcpy(cudaCache, ch, sizeof(struct cache), cudaMemcpyHostToDevice);
	//if (err != cudaSuccess) printf("%s memcpy2\n", cudaGetErrorString(err));
	if(balance > 0.0) {
	compute_image <<<blockCount, blockSize>>>(cudaCoords, width, height, colorsSet, cudaCache, blockCount, blockSize, balance);
	}
	compute_imageCPU(coords, width, height, c, balance);
	if(balance > 0.0) {
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("%s synch\n", cudaGetErrorString(err));

	err = cudaMemcpy(c, colorsSet, n * balance * sizeof(struct colors), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printf("%s memcpy3\n", cudaGetErrorString(err));

	//err = cudaMemcpy(ch, cudaCache, sizeof(struct cache), cudaMemcpyDeviceToHost);
	//if (err != cudaSuccess) printf("%s memcpy4\n", cudaGetErrorString(err));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("%s synch2\n", cudaGetErrorString(err));
	}
	clock_gettime(CLOCK_MONOTONIC, &endTime);
	runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	#ifdef BENCHMARK
	//get metadata to print
	printf("Blocks: %d\tThreads per Block: %d\tSize:%dx%d\tDepth: %d\tTime: %f\n",
	blockCount, blockSize, width, height, coords->maxiter, runTime);
	#else
	fprintf(stderr, "\ncalculating frame took %lf seconds\n", runTime);
	#endif
	
	
	#ifndef NOX
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			
			draw_point(i, j, c[i * width + j]);
			
		}
	#endif
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
	windowWidth = 640;
	windowHeight = 480;
	coordSet* dispCoords = (coordSet*)malloc(sizeof(coordSet));
	
	if(argv[1] && argv[2] && argv[3] && argv[4]){
		dispCoords->xmin = atof(argv[1]);
		dispCoords->xmax = atof(argv[2]);
		dispCoords->ymin = atof(argv[3]);
		dispCoords->ymax = atof(argv[4]);
		//dispCoords->maxiter = atoi(argv[5]);
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

	if(argc>9)
	{
		dispCoords->maxiter = atoi(argv[5]);
		windowWidth = atoi(argv[6]);
		windowHeight = atoi(argv[7]);
		blockCount = atoi(argv[8]);
		blockSize = atoi(argv[9]);
	}

	#ifndef NOX
	// Open a new window.
	if(windowWidth < 2048)
	gfx_open(windowWidth,windowHeight,"Mandelbrot Fractal");
	else {
		gfx_open(512,512,"Placeholder for Benchmark Only");
		printf("You have chosen a window size greater than 2048. You will not see a visualization, but the benchmark is running and will calculate results shortly.\n");
	}


	// Fill it with a dark blue initially.
	gfx_clear_color(0,0,255);
	gfx_clear();
	#endif
	//draw intial position
	reDraw(dispCoords);

	#ifndef NOX
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
	#endif

	return 0;
}
