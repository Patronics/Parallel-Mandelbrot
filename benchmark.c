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
	printf("\t\tapproach #1: serial implementation\n");
	printf("\t\tapproach #2: OpenMP implementation\n");
	printf("\t\tapproach #3: final CUDA implementation\n");
	//TODO:
	//printf("approach #4 to ... : intermediate CUDA implementation, with optimizations x, y z\n");
	exit(1);
}

int main(int argc, char *argv[]){
	char** approaches = malloc((NUM_APPROACHES+1) * sizeof(char*));
	approaches[0] = "echo"; //for testing, secret approach 0
	approaches[1] = "./serialfractal";
	approaches[2] = "./fractal";
	approaches[3] = "./cudafractal";
	
	if(argc < 2){
		usage();
	}
	
	//because just used as arugments anyway, can just keep variables as strings
	char* n = "512";
	char* m = "512";
	//n and m are unused in non-cuda implementations
	char* dim = "1600";
	char* max_iter = "100";
	
	int approach_number = 1;
	
	if(argc == 2){
		approach_number = atof (argv[1]);
	}
	if(argc >= 4){
		n = argv[1];
		m = argv[2];
		if( argc == 4 && atof (argv[3])<NUM_APPROACHES){
			approach_number = atof (argv[3]);
		}else{
			dim = argv[3];
		}
	}
	if(argc >= 5){
		if( argc == 5 && atof (argv[4])<NUM_APPROACHES){
			approach_number = atof (argv[4]);
		}else{
			max_iter = argv[4];
		}
	}
	if(argc >= 6){
		approach_number = atof (argv[5]);
	}
	
	if (approach_number > NUM_APPROACHES){
		usage();
	}
	execlp(approaches[approach_number],approaches[approach_number], "-1.0", "1.0","-1.0","1.0", max_iter, dim, dim, n, m, NULL);
	
}


