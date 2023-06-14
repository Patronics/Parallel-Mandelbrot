#!/usr/bin/env python3
import sys
import subprocess

if __name__ == "__main__":
	if(len(sys.argv)>=2):
		for i in range(int(sys.argv[1])):
			processInfo = subprocess.run(["./benchmark"]+sys.argv[2:],capture_output=True)
			if(i == 0): #print full details only on first iteration of loop
				print(f"{processInfo.stdout.decode().strip()}:")
			if(i == int(sys.argv[1])-1):
				print(processInfo.stdout.decode().split("Time: ")[1].strip(), end='')
			else:
				print(processInfo.stdout.decode().split("Time: ")[1].strip(), end=',')
	else:
		print(f"usage: {sys.argv[0]} timesToIterate argsForBenchmark*")
		
	print("")