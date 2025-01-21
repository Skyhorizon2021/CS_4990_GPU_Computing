#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}
//compute vector sum z_h = x_h + y_h
void vecAdd_h(float* x_h, float* y_h, float* z_h, unsigned int n)
{
    for(unsigned int i=0; i<n; i++)
    {
        z_h[i] = x_h[i] + y_h[i];
    }
}

int main(int argc, char** argv)
{
    double startTime = myCPUTimer();
    unsigned int n = 16777216;

    //allocate host memory for x_h, y_h, and z_h and intialize x_h, y_h
    
    float* x_h = (float*) malloc(sizeof(float)*n);
    for(unsigned int i = 0; i < n; i++) x_h[i] = (float) rand()/(float)(RAND_MAX);
    float* y_h = (float*) malloc(sizeof(float)*n);
    for(unsigned int i = 0; i < n; i++) y_h[i] = (float) rand()/(float)(RAND_MAX);
    float* z_h = (float*) calloc(n, sizeof(float));
    
    //Perfom CPU vector addition
    
    vecAdd_h(x_h,y_h,z_h,n);
    double endTime = myCPUTimer();
    printf("vecAdd on CPU: %f s\n", endTime - startTime);
    
    // free host memory of x_h, z_h, and z_h
    free(x_h);
    free(y_h);
    free(z_h);

    return 0;
}