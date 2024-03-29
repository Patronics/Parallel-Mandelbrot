#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define NUM_APPROACHES 5

void usage(){
	printf("Usage: benchmark [n] [m] [dim] [max_iter] [approach_number]\n");
	printf("\tn\t\t=\tnumber of blocks (defaults to 512)\n");
	printf("\tm\t\t=\tthreads per block (defaults to 512)\n");
	printf("\tdim\t\t=\twidth/height of canvas in pixels (defaults to 1600)\n");
	printf("\tmax_iter\t=\tmax iterations (defaults to 100)\n\n");
	printf("\tapproach_number\t=\tapproach number to demonstrate, (defaults to 1, acceptable range from 1 to %d)\n\n", NUM_APPROACHES);
	printf("\t\tapproach #1: serial implementation\n");
	printf("\t\tapproach #2: OpenMP implementation\n");
	printf("\t\tapproach #3: CUDA implementation with caching\n");
	printf("\t\tapproach #4: CUDA/CPU load balancing\n");
	printf("\t\tapproach #5: CUDA with smart calculations\n");
	//TODO:
	//printf("approach #4 to ... : intermediate CUDA implementation, with optimizations x, y z\n");
	exit(1);
}

int main(int argc, char *argv[]){
	char** approaches = malloc((NUM_APPROACHES+1) * sizeof(char*));
	approaches[0] = "echo"; //for testing, secret approach 0
	approaches[1] = "./serialfractal";
	approaches[2] = "./fractal";
	approaches[3] = "./cudafractal_cache";
	approaches[4] = "./cudafractal_loadbalance";
	approaches[5] = "./cudafractal_loopbreak";
	char * approachSuffix = "";
	//check if running "Xbenchmark" or not
	if(strchr(argv[0], 'X') == NULL){
		approachSuffix = "-nox";
	}
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
	char approach[32];
	strcpy(approach, approaches[approach_number]);
	strcat(approach, approachSuffix);
	execlp(approach,approach, "-1.5", "0.5","-1.0","1.0", max_iter, dim, dim, n, m, NULL);
	
}


