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
#include <omp.h>

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
	uint32_t r;
	uint32_t g;
	uint32_t b;
};

void initColors(uint32_t* colorArray)
{
	for(int i = 0; i < 3001; i++)
	{
		colorArray[i] = (255 * i / 3000) << 16;
		colorArray[i] += (((255 * i / 100) & 0xFF) << 8);
		colorArray[i] += ((255 * i / 30) & 0xFF);
	}
	printf("%d\n", colorArray[3000]);
}
/*
Compute the number of iterations at point x, y
n the complex space, up to a maximum of maxiter.
Return the number of iterations at that point.

This example computes the Mandelbrot fractal:
z = z^2 + alpha

Where z is initially zero, and alpha is the location x + iy
in the complex plane.  Note that we are using the "complex"
numeric type in C, which has the special functions cabs()
and cpow() to compute the absolute values and powers of
complex values.
*/

__device__ uint16_t compute_point( double x, double y, uint16_t max )
{
	double z_real = 0;
	double z_imaginary = 0;
	double z_realsquared = 0;
	double z_imaginarysquared = 0;

	uint16_t iter = 0;
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
uint16_t compute_pointCPU( double x, double y, uint16_t max )
{
        double z_real = 0;
        double z_imaginary = 0;
        double z_realsquared = 0;
        double z_imaginarysquared = 0;

        uint16_t iter = 0;
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
/*
Compute an entire image, writing each point to the given bitmap.
Scale the image to the range (xmin-xmax,ymin-ymax).
*/

__global__ void compute_image(double xmin, double ymin, int maxiter, int width, int height, uint32_t* colorsArray, uint32_t* cols, double stepFactorX, double stepFactorY)
{
	//double xmin=coords->xmin;
	//double xmax=coords->xmax;
	//double ymin=coords->ymin;
	//double ymax=coords->ymax;
	//int maxiter=coords->maxiter;

    uint32_t my_i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t my_j = blockDim.y * blockIdx.y + threadIdx.y;
	//int total_threads = gridDim.x * blockDim.x;
	//int total_threads1 = gridDim.y * blockDim.y;
	if(my_i < width && my_j < height) {	
		
    double x = xmin + my_i*stepFactorX;
	double y = ymin + my_j*stepFactorY;

    uint16_t iter = compute_point(x,y,maxiter);
    //colorsSet[my_i+width*my_j].r = 255 * iter / maxiter;
	//colorsSet[my_i+width*my_j].g = 255 * iter / (maxiter/30);
	//colorsSet[my_i+width*my_j].b = 255 * iter / (maxiter/100);
    	//colorsSet[my_i+width*my_j].r = (colorsArray[iter] & 0xFF0000) >> 16;
	//colorsSet[my_i+width*my_j].b = (colorsArray[iter] & 0xFF00) >> 8;
	//colorsSet[my_i+width*my_j].g = colorsArray[iter] & 0xFF;
	cols[my_i+width*my_j] = colorsArray[iter];
	}
}

void compute_imageCPU(coordSet* coords, int width, int height, struct colors *colorsSet, uint32_t* colorsArray, uint32_t* cols)
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
        for(int i = height*0.6; i < height; ++i) {

		#pragma omp parallel for schedule(dynamic)
		for(int j = 0; j < width; ++j) {

    double x = xmin + j*(xmax-xmin)/width;
        double y = ymin + i*(ymax-ymin)/height;

    uint16_t iter = compute_pointCPU(x,y,maxiter);
    //colorsSet[i*width+j].r = 255 * iter / maxiter;
        //colorsSet[i*width+j].g = 255 * iter / (maxiter/30);
        //colorsSet[i*width+j].b = 255 * iter / (maxiter/100);
	//colorsSet[i*width+j].r = (colorsArray[iter] & 0xFF0000) >> 16;
        //colorsSet[i*width+j].b = (colorsArray[iter] & 0xFF00) >> 8;
        //colorsSet[i*width+j].g = colorsArray[iter] & 0xFF;
    	cols[i*width+j] = colorsArray[iter];
        }
	}
	double end_time = omp_get_wtime();
        printf("%.5f\n", end_time - start_time);
}


void draw_point(int i, int j, struct colors c,uint32_t* cols)
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
    //int width = gfx_xsize();
	//int height = gfx_ysize();
	int width = 18000;
	int height = 18000;
    int n = width * height;
	
	//TODO bigger blocks are likely faster
	#define BLOCK_SIZE 16
    	double xMin = coords->xmin;
	double yMin = coords->ymin;
	double xStep = (coords->xmax-coords->xmin)/width;
	double yStep = (coords->ymax-coords->ymin)/height;
	double maxIter = coords->maxiter;
	dim3 dimBlock(16, 16); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	dim3 dimGrid(width/BLOCK_SIZE, height/BLOCK_SIZE); // 1*1 blocks in a grid
	uint32_t* colorsArray = (uint32_t*) malloc(3001 * sizeof(uint32_t));
	//struct colors* colorsSet;
	//struct colors* c = (struct colors*)malloc(n * sizeof(struct colors));
	uint32_t* cols = (uint32_t*)malloc(n*sizeof(uint32_t));
	uint32_t* cudacols;
	cudaMalloc(&cudacols, n * sizeof(uint32_t));
	initColors(colorsArray);
	//cudaMalloc(&colorsSet, n * sizeof(struct colors));
	//coordSet* cudaCoords;
	//cudaMalloc(&cudaCoords, sizeof(coordSet));
	uint32_t* cuda_array;
	cudaMalloc(&cuda_array, 3001 * sizeof(uint32_t));
	// Show the configuration, just in case you want to recreate it.
	//printf("coordinates: %lf %lf %lf %lf\n",coords->xmin,coords->xmax,coords->ymin,coords->ymax);
	// Display the fractal image

	struct timespec startTime, endTime;
	double runTime;
	//clock_gettime(CLOCK_MONOTONIC, &startTime);
	// this is not the actual block size and thread count
	cudaError_t whatever = cudaMemcpy(cuda_array, colorsArray, 3001 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	//cudaError_t err = cudaMemcpy(cudaCoords, coords,sizeof(coordSet), cudaMemcpyHostToDevice);
	//if (err != cudaSuccess) printf("%s memcpy0 coords\n", cudaGetErrorString(err));
	//err = cudaMemcpy(colorsSet, c, n * sizeof(struct colors), cudaMemcpyHostToDevice);
	cudaMemcpy(cudacols,cols,n*sizeof(uint32_t),cudaMemcpyHostToDevice);
	//if (err != cudaSuccess) printf("%s memcpy1\n", cudaGetErrorString(err));
	double start = omp_get_wtime();
	compute_image <<<dimGrid, dimBlock>>>(xMin,yMin,maxIter, width, height,cuda_array,cudacols,xStep,yStep);
	//double end = omp_get_wtime();
	//compute_imageCPU(coords, width, height, c,colorsArray,cols);
	//err = cudaDeviceSynchronize();
	//double end = omp_get_wtime();
	//if (err != cudaSuccess) printf("%s synch\n", cudaGetErrorString(err));
	//err = cudaMemcpy(c, colorsSet, n * sizeof(struct colors), cudaMemcpyDeviceToHost);
	cudaError_t err = cudaMemcpy(cols,cudacols, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//if (err != cudaSuccess) printf("%s memcpy2\n", cudaGetErrorString(err));
	//err = cudaDeviceSynchronize();
	
	//clock_gettime(CLOCK_MONOTONIC, &endTime);
	//runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	//fprintf(stderr, "calculating frame took %lf seconds\n", runTime);
	double end = omp_get_wtime();
	printf("time: %lf\n", end - start);	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			//c[i * width + j].r=j;
			uint32_t r = (cols[i*width+j] & 0xFF0000) >> 16;
		       	uint32_t g = (cols[i*width+j] & 0xFF00) >> 8;
			uint32_t b = (cols[i*width+j] & 0xFF);	
			        gfx_color(r,g,b);
        			// Plot the point on the screen.
        			gfx_point(j, i);
		}
	
	//clock_gettime(CLOCK_MONOTONIC, &endTime);
	//runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	//fprintf(stderr, "\ncalculating and rendering frame took %lf seconds\n", runTime);

	//free(c);
	//cudaFree(colorsSet);
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


int main( int argc, char *argv[] ){
	// The initial boundaries of the fractal image in x,y space.
	const double xminDefault = -2.0;
	const double xmaxDefault = 1.0;
	const double yminDefault = -1.5;
	const double ymaxDefault=  1.5;
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
		setMidpoints(dispCoords);
	}else{
		dispCoords->xmin=xminDefault;
		dispCoords->xmax=xmaxDefault;
		dispCoords->ymin=yminDefault;
		dispCoords->ymax=ymaxDefault;
		dispCoords->maxiter=maxiterDefault;
		setMidpoints(dispCoords);
	}


	// Open a new window.
	gfx_open(640,640,"Mandelbrot Fractal");


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
