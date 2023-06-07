#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_APPROACHES 3

void usage(){
	printf("Usage: benchmark [n] [m] [dim] [max_iter] [approach_number]\n");
	printf("\tn\t\t=\tnumber of blocks (defaults to 512)\n");
	printf("\tm\t\t=\tthreads per block (defaults to 512)\n");
	printf("\tdim\t\t=\twidth/height of canvas in pixels (defaults to 1600)\n");
	printf("\tmax_iter\t=\tmax iterations (defaults to 100)\n\n");
	printf("\tapproach_number\t=\tapproach number to demonstrate, (defaults to 1, acceptable range from 1 to %d)\n\n", NUM_APPROACHES);
	exit(1);
}

int main(int argc, char *argv[]){
	char** approaches = malloc((NUM_APPROACHES+1) * sizeof(char*));
	approaches[0] = "echo"; //for debug, secret approach 0
	approaches[1] = "./serialfractal";
	approaches[2] = "./fractal";
	approaches[3] = "./cudafractal";
	
	if(argc < 2){
		usage();
	}
	
	//because just used as arugments anyway, can just keep them as strings
	char* n = "512";
	char* m = "512";
	char* dim = "1600";
	char* max_iter = "3000";
	int approach_number = 1;
	
	if(argc == 2){
		approach_number = atof (argv[1]);
	}
	
	
	execlp(approaches[approach_number],approaches[approach_number], "-1.5", "0.5","-1.0","1.0", max_iter, dim, dim, NULL);
	
}


