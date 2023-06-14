/*
fractal.c - Parallel interactive Mandelbrot Fractal Display
based on starting code for CSE 30341 Project 3.
*/

#define BENCHMARK


#ifndef NOX
#include "gfx.h"
#endif


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>

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


#ifdef NOX
//can't get dimensions from window if not using x, so share as global instead
	int global_width=0;
	int global_height=0;
#endif




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

static int compute_point( double x, double y, int max )
{
	double complex z = 0;
	double complex alpha = x + I*y;
	
	#ifdef CARDIOID
	if ((cabs(1 - csqrt(1 - 4 * alpha)) <= 1) || (cabs(1 + alpha) <= 0.25))
                return max;
	#endif

	int iter = 0;

	if(y==0){
		while( cabs(z)<4 && iter < max ) {
			z = cpow(z,2) + alpha;
			iter++;
		}
	} else {
		while( cabs(z)<4 && iter < max ) {
			z = z*z + alpha;
			iter++;
		}
	}

	return iter;
}

/*
Compute an entire image, writing each point to the given bitmap.
Scale the image to the range (xmin-xmax,ymin-ymax).
*/

void compute_image(coordSet* coords)
{
	int i,j;
	double xmin=coords->xmin;
	double xmax=coords->xmax;
	double ymin=coords->ymin;
	double ymax=coords->ymax;
	int maxiter=coords->maxiter;

	#ifndef NOX
	int width = gfx_xsize();
	int height = gfx_ysize();
	#else
	int width = global_width;
	int height = global_height;
	#endif

	// For every pixel i,j, in the image...
	#pragma omp parallel for schedule(dynamic)
	for(j=0;j<height;j++) {
		#pragma omp parallel for schedule(dynamic)
		for(i=0;i<width;i++) {
			// Scale from pixels i,j to coordinates x,y
			double x = xmin + i*(xmax-xmin)/width;
			double y = ymin + j*(ymax-ymin)/height;

			// Compute the iterations at x,y
			int iter = 0;
			//#pragma omp critical
			iter = compute_point(x,y,maxiter);

			// Convert a iteration number to an RGB color.
			// (Change this bit to get more interesting colors.)
			//int gray = 255 * iter / maxiter;
			
			//volatile prevents the compiler from optimizing these variables away in the no-x version
			#ifdef NOX
			volatile int r = 255 * iter / maxiter;
			volatile int g = 255 * iter / (maxiter/30);
			volatile int b = 255 * iter / (maxiter/100);
			
			#else
			int r = 255 * iter / maxiter;
			int g = 255 * iter / (maxiter/30);
			int b = 255 * iter / (maxiter/100);
			#pragma omp critical (plotpixel)
			{
				gfx_color(r,g,b);
				// Plot the point on the screen.
				gfx_point(i,j);
			}
			#endif
		}
	}
}

void setMidpoints(coordSet* coords){
	coords->xmid = (coords->xmin+coords->xmax)/2;
	coords->ymid = (coords->ymin+coords->ymax)/2;

}

void reDraw(coordSet* coords){
	// Show the configuration, just in case you want to recreate it.
	#ifndef BENCHMARK
	printf("coordinates: %lf %lf %lf %lf\n",coords->xmin,coords->xmax,coords->ymin,coords->ymax);
	#endif
	// Display the fractal image


	struct timespec startTime, endTime;
	double runTime;
	clock_gettime(CLOCK_MONOTONIC, &startTime);

	compute_image(coords);

	clock_gettime(CLOCK_MONOTONIC, &endTime);
	runTime = difftime(endTime.tv_sec, startTime.tv_sec)+((endTime.tv_nsec-startTime.tv_nsec)/1e9);
	#ifdef BENCHMARK
	//get metadata to print
		#ifndef NOX
		int width = gfx_xsize();
		int height = gfx_ysize();
		#else
		int width = global_width;
		int height = global_height;
		#endif
		int numThreads = 1;
		#ifdef OPENMP
		#pragma omp parallel
		{
			#pragma omp single
			{
				numThreads = omp_get_num_threads();
			}
		}
		#endif
	printf("Blocks: %d\tThreads per Block: %d\tSize:%dx%d\tDepth: %d\tTime: %f\n",
	1, numThreads, width, height, coords->maxiter, runTime);
	#else
	fprintf(stderr, "\nrendering frame took %lf seconds\n", runTime);
	#endif
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
	const double xminDefault = -1.5;
	const double xmaxDefault = 0.5;
	const double yminDefault = -1.0;
	const double ymaxDefault=  1.0;
	// Maximum number of iterations to compute.
	// Higher values take longer but have more detail.
	const int maxiterDefault = 3000; //default 500
	int windowWidth = 640;
	int windowHeight = 480;
	
	coordSet* dispCoords = malloc(sizeof(coordSet));
	
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
	
	if(argc>7){
		windowWidth = atof(argv[6]);
		windowHeight = atof(argv[7]);
	}

	#ifdef NOX
	
	global_width = windowWidth;
	global_height = windowHeight;
	
	#else
	// Open a new window.
	gfx_open(windowWidth,windowHeight,"Mandelbrot Fractal");

	// Fill it with a dark blue initially.
	gfx_clear_color(0,0,255);
	gfx_clear();
	#endif

	//draw intial position
	reDraw(dispCoords);

	//main loop, only if X window initialized
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
