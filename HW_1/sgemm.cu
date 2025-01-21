#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

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

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m) 
    {
        for (int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= n * n)
        {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if (idx >= n * n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0; // Use int type for tmp
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    srand(3333);
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // Allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int) * m * n));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int) * n * k));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int) * m * k));
    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int) * m * k));

    // Random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 100;
        }
    }

    // Random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 100;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // Events to count the execution time
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Start to count execution time of GPU version
    CHECK(cudaEventRecord(start, 0));
    
    // Allocate memory space on the device 
    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int) * m * n));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int) * n * k));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int) * m * k));

    // Copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // Launch kernel 
    if (m == n && n == k)
    {
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
    }
    
    CHECK(cudaDeviceSynchronize());

    // Transfer results from device to host 
    CHECK(cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost));

    // Time counting terminate
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // Compute time elapsed on GPU computing
    CHECK(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    printf("%dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

    // Start the CPU version
    CHECK(cudaEventRecord(start, 0));

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop));
    printf("%dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

    // Validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (h_cc[i * k + j] != h_c[i * k + j])
            {
                all_ok = 0;
                break;
            }
        }
        if (!all_ok) break;
    }

    // Compute speedup
    if (all_ok)
    {
        printf("Speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("Incorrect results\n");
    }

    // Free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    CHECK(cudaFreeHost(h_cc));

    return 0;
}
