# tf_test_gpu

This is a simple test script to check that the GPU is operating as expected for Tensorflow. 
Its is based on a script from https://learningtensorflow.com/lesson10/.

The original script has been modified for python 3 along with some refinements.

This script sets up a matrix defined in the command line arguments and performs a matrix mulitplication and dot product
and outputs the result to screen along with the time and device used to perform the operation.
This script can be run as a quick sanity check as part of another project.

## Dependencies 
- sys module
- os module
- numpy
- tensorflow (tensorflow_gpu)
- datetime module

## useage

The script requires to arguments as shown below

python test_gpu.py gpu 1000

This set the gpu up to run a 1000 x 1000 matrix operation. You could substitute in cpu for gpu.

Its worth noting that for small matrix operations the CPU will be quicker, as the io overhead limits the
performance of the GPU but as the matrix (tehsor) size grows the gpu will be quicker.
