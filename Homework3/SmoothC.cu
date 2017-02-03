//#include <iostream>
#include <cmath>
#include <cuda.h>

//using std::cout;
//using std::endl;

//function is run on the device; specify device functions before global functions
__device__ float myAbs(float value) {
   return (value < 0) ? (value * -1) : value;
} //end abs function

__global__ void findVals(float *x, float *y, int n, float h, float *sum, int *count){
//kernel: compute the sum and count - used later to compute mean

   int me = blockIdx.x*blockDim.x+threadIdx.x;

   float xi = x[me];
   sum[me] = 0;
   count[me] = 0;

   for(int j=0; j<n; j++){
       //now iterate through the j values
       float xj = x[j]; //not needed - used for better visibility
       if(myAbs(xj-xi) < h){
          sum[me] += y[j];
          count[me]++;
       } //end abs if condition
   } //end j for loop
} //end findVals function

//this is out "main" function - handles all the computation
void smoothc(float *x, float *y, float *m, int n, float h){
  float *devx, //device x
        *devy, //device y
        *hsum, //host sums
        *dsum; //device sum
  int *hcount, //host count
      *dcount; //device count

  //size of arrays in bytes
  int floatSize = n*sizeof(float);
  int intSize = n*sizeof(int);

  //allocate space on host
  hsum = (float *) malloc(floatSize);
  hcount = (int *) malloc(intSize);

  //allocate space on device
  cudaMalloc((void **)&dsum, floatSize);
  cudaMalloc((void **)&dcount, intSize);
  cudaMalloc((void **)&devx, floatSize);
  cudaMalloc((void **)&devy, floatSize);

  //Copy host parameters for device
  cudaMemcpy(devx,x,floatSize,cudaMemcpyHostToDevice);
  cudaMemcpy(devy,y,floatSize,cudaMemcpyHostToDevice);

  int nThreads = min(n,500),
      nBlocks = ceil(n/nThreads);

  dim3 dimGrid(nBlocks,1);
  dim3 dimBlock(nThreads,1,1);

  //invoke the kernel
  findVals<<<dimGrid,dimBlock>>>(devx, devy, n, h, dsum, dcount);
  //wait for kernel to finish;
  cudaThreadSynchronize();
  //copy results: device to host
  cudaMemcpy(hsum,dsum,floatSize,cudaMemcpyDeviceToHost);
  cudaMemcpy(hcount,dcount,intSize,cudaMemcpyDeviceToHost);

  //compute the means
  for(int i=0; i<n; i++){
      m[i] = hsum[i]/hcount[i];
      //NOTE: assuming perfect precision, hcount[i] must always be >=1 since |xi-xi|<h for all h>0
  } //end i for loop

  //clean up
  free(hsum);
  cudaFree(dsum);
  free(hcount);
  cudaFree(dcount);
  cudaFree(devx);
  cudaFree(devy);
} //end smoothc function

/*
int main() {

    int n = 10;
    float x[n];
    float y[n];
    float xcount = 10;
    float h = 0.1;
    for(int i = 0; i < n; i++) {
        x[i] = y[i] = xcount;
    }

   //return array
   float m[n];

   smoothc(x, y, m, n, h);

   for(int i = 0; i < n; i++) {
       cout<<m[i]<<" ";
   } cout<<endl;
   
}
*/
