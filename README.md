# Parallel Mandelbrot

## Compiling and Running

run `make` to build all scripts

run the OpenMP version with `./fractal`, or the serial version with `./serialfractal`

Run benchmarking versions with `./benchmark` or `./Xbenchmark` (they will print usage details when run with no arguments)

## Controls

|Key                        |function              |modifier                | modified function|
|---------------------------|----------------------|------------------------|------------------|
|**q**                      |**q**uit              |                        |                  |
|**i**/**o**                |zoom **i**n/**o**ut   |**I**/**O**             | zoom smoothly    |
|**w**/**a**/**s**/**d**    |pan up/left/down/right|**W**/**A**/**S**/**D** | move smoothly    |
|**r**                      |**r**eset default view|                        |                  |
|**R**                      |**R**eflect view      |                        |                  |




## resources
- uses the [gfx library](https://www3.nd.edu/~dthain/courses/cse30341/spring2020/project3/gfx) from the University of Notre Dame

- also based on their [starting code for rendering the mandelbrot set](https://www3.nd.edu/~dthain/courses/cse30341/spring2020/project3/)
