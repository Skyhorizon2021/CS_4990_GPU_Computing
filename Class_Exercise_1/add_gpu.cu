#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda_runtime.h>

#define CHECK(call)                                           \
{                                                             \
    const cudaError_t error = call;                           \
    if (error != cudaSuccess)                                 \
    {                                                         \
        printf("Error: %s:%d, ", __FILE__, __LINE__);         \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                              \
    }                                                         \
}
__global__ void vecAddKernel(float* x_d, float* y_d, float* z_d, unsigned int n) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i < n) // handling boundary conditions
    {
        z_d[i] = x_d[i] + y_d[i];
    }

}
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

int main(int argc, char** argv){
    double vecAddstart = myCPUTimer();
    unsigned int n = 16777216;

    //allocate host memory for x_h, y_h, and z_h and intialize x_h, y_h
    float* x_h = (float*) malloc(sizeof(float)*n);
    for(unsigned int i = 0; i < n; i++) x_h[i] = (float) rand()/(float)(RAND_MAX);
    float* y_h = (float*) malloc(sizeof(float)*n);
    for(unsigned int i = 0; i < n; i++) y_h[i] = (float) rand()/(float)(RAND_MAX);
    float* z_h = (float*) calloc(n, sizeof(float));

    // (1) allocate device memory for arrays x_d, y_d, z_d
    float *x_d, *y_d, *z_d;
    double startTime = myCPUTimer();
    cudaMalloc((void**) &x_d, sizeof(float)*n);
    cudaMalloc((void**) &y_d, sizeof(float)*n);
    cudaMalloc((void**) &z_d, sizeof(float)*n);
    double endTime = myCPUTimer();
    printf("\t cudaMalloc: \t\t\t\t%f s\n",endTime - startTime);
    // (2) copy arrays x_h and y_h to device memory x_d and y_d, respectively
    startTime = myCPUTimer();
    cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    endTime = myCPUTimer();
    printf("\t cudaMemcpy: \t\t\t\t%f s\n",endTime - startTime);
    // (3) call kernel to launch a grid of threads to perform the vector addition on GPU
    startTime = myCPUTimer();
    vecAddKernel<<<ceil(n/512.0), 512>>>(x_d, y_d, z_d, n);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\t vecAddKernel<<<(%d,1,1),(512,1,1)>>>: %f s\n", (n + 511) / 512, endTime - startTime);

    // (4) Copy the result data from device memory of array  z_d to host memory of array z_h
    startTime = myCPUTimer();
    cudaMemcpy(z_h, z_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
    endTime = myCPUTimer();
    printf("\t cudaMemcpy: \t\t\t\t%f s\n",endTime - startTime);


    // Total time for vecAdd on GPU 
    double vecAddend = myCPUTimer();
    printf("vecAdd on GPU: %f s\n", vecAddend - vecAddstart);
    // (5) free device memory of x_d, y_d, and z_d 
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

    // free host memory of x_h, z_h, and z_h
    free(x_h);
    free(y_h);
    free(z_h);

    return 0;
}

