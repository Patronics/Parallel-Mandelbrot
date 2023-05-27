/*
fractal.c - Parallel interactive Mandelbrot Fractal Display
based on starting code for CSE 30341 Project 3.
*/

#include "gfx.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>

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

	int iter = 0;

	while( cabs(z)<4 && iter < max ) {
		z = cpow(z,2) + alpha;
		iter++;
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

	int width = gfx_xsize();
	int height = gfx_ysize();

	// For every pixel i,j, in the image...
	#pragma omp parallel for
	for(j=0;j<height;j++) {
		#pragma omp parallel for
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
			int r = 255 * iter / maxiter;
			int g = 255 * iter / (maxiter/30);
			int b = 255 * iter / (maxiter/100);
			#pragma omp critical (plotpixel)
			{
				gfx_color(r,g,b);
				// Plot the point on the screen.
				gfx_point(i,j);
			}
		}
	}
}

void setMidpoints(coordSet* coords){
	coords->xmid = (coords->xmin+coords->xmax)/2;
	coords->ymid = (coords->ymin+coords->ymax)/2;

}

void reDraw(coordSet* coords){
	// Show the configuration, just in case you want to recreate it.
	printf("coordinates: %lf %lf %lf %lf\n",coords->xmin,coords->xmax,coords->ymin,coords->ymax);
	// Display the fractal image
	compute_image(coords);
}


void zoomIn(coordSet* coords){
	setMidpoints(coords);
	coords->xmax=(coords->xmid+coords->xmax)/2;
	coords->xmin=(coords->xmid+coords->xmin)/2;
	coords->ymax=(coords->ymid+coords->ymax)/2;
	coords->ymin=(coords->ymid+coords->ymin)/2;
	setMidpoints(coords);
	reDraw(coords);
}

void zoomOut(coordSet* coords){
setMidpoints(coords);
coords->xmax=coords->xmax+(coords->xmax-coords->xmid)*4;
coords->xmin=coords->xmin+(coords->xmin-coords->xmid)*4;
coords->ymax=coords->ymax+(coords->ymax-coords->ymid)*4;
coords->ymin=coords->ymin+(coords->ymin-coords->ymid)*4;
setMidpoints(coords);
reDraw(coords);
}

/*
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
*/

int main( int argc, char *argv[] )
{
	// The initial boundaries of the fractal image in x,y space.
	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

	// Maximum number of iterations to compute.
	// Higher values take longer but have more detail.
	int maxiter=3000; //default 500
	coordSet* dispCoords = malloc(sizeof(coordSet));
	dispCoords->xmin=xmin;
	dispCoords->xmax=xmax;
	dispCoords->ymin=ymin;
	dispCoords->ymax=ymax;
	dispCoords->maxiter=maxiter;
	setMidpoints(dispCoords);
	// Open a new window.
	gfx_open(640,480,"Mandelbrot Fractal");


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
			exit(0);
		case 'i':
			printf("zooming in\n");
			zoomIn(dispCoords);
			break;
		case 'o':
			printf("zooming out\n");
			zoomOut(dispCoords);
		}
//		} else if(c=='q'){
	}

	return 0;
}
