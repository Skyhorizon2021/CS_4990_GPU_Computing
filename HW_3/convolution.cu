#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// Timer function for CPU
double myCPUTimer() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

// Kernel configuration
#define TILE_SIZE 16
#define FILTER_WIDTH 5

// Average filter for constant memory
__constant__ float d_filter[FILTER_WIDTH * FILTER_WIDTH];

// Host function prototypes
void blurImage_h(cv::Mat &Pout_Mat_h, cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols);
void blurImage_d(cv::Mat &Pout_Mat_h, cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols);
void blurImage_tiled_d(cv::Mat &Pout_Mat_h, cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols);
bool verify(cv::Mat &answer1, cv::Mat &answer2, unsigned int nRows, unsigned int nCols);

// Kernel prototypes
__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height);
__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height);

// Main function
int main(int argc, char **argv) {
    // Read input image
    cv::Mat inputImage = cv::imread("santa-grayscale.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error reading the image." << std::endl;
        return -1;
    }

    unsigned int nRows = inputImage.rows;
    unsigned int nCols = inputImage.cols;

    cv::Mat outputImage_CPU = inputImage.clone();
    cv::Mat outputImage_GPU = inputImage.clone();
    cv::Mat outputImage_Tiled = inputImage.clone();
    cv::Mat referenceBlur;

    // Reference using OpenCV's cv::blur
    double startTime = myCPUTimer();
    cv::blur(inputImage, referenceBlur, cv::Size(FILTER_WIDTH, FILTER_WIDTH));
    double endTime = myCPUTimer();
    printf("Blur with OpenCV: %f s\n",endTime-startTime);
    // Perform CPU blur
    startTime = myCPUTimer();
    blurImage_h(outputImage_CPU, inputImage, nRows, nCols);
    endTime = myCPUTimer();
    printf("Blur with CPU: %f s\n",endTime-startTime);
    // Perform GPU blur with no tiling
    startTime = myCPUTimer();
    blurImage_d(outputImage_GPU, inputImage, nRows, nCols);
    endTime = myCPUTimer();
    printf("Blur with GPU and no tiling: %f s\n",endTime-startTime);
    // Perform GPU blur with tiling
    startTime = myCPUTimer();
    blurImage_tiled_d(outputImage_Tiled, inputImage, nRows, nCols);
    endTime = myCPUTimer();
    printf("Blur with GPU and tiling: %f s\n",endTime-startTime);

    // Verify results
    if (verify(outputImage_CPU, referenceBlur, nRows, nCols))
        std::cout << "CPU blur matches OpenCV reference." << std::endl;
    if (verify(outputImage_GPU, referenceBlur, nRows, nCols))
        std::cout << "GPU without tiled blur matches OpenCV reference." << std::endl;
    if (verify(outputImage_Tiled, referenceBlur, nRows, nCols))
        std::cout << "GPU tiled blur matches OpenCV reference." << std::endl;

    cv::imwrite("blurredImg_opencv.jpg", referenceBlur);
    cv::imwrite("blurredImg_cpu.jpg", outputImage_CPU);
    cv::imwrite("blurredImg_gpu.jpg", outputImage_GPU);
    cv::imwrite("blurredImg_tiled_gpu.jpg", outputImage_Tiled);

    return 0;
}

void blurImage_h(cv::Mat &Pout_Mat_h, cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            float sum = 0.0f;
            for (int di = -FILTER_WIDTH / 2; di <= FILTER_WIDTH / 2; di++) {
                for (int dj = -FILTER_WIDTH / 2; dj <= FILTER_WIDTH / 2; dj++) {
                    int ni = min(max(i + di, 0), nRows - 1);
                    int nj = min(max(j + dj, 0), nCols - 1);
                    sum += Pin_Mat_h.at<unsigned char>(ni, nj);
                }
            }
            Pout_Mat_h.at<unsigned char>(i, j) = static_cast<unsigned char>(sum / (FILTER_WIDTH * FILTER_WIDTH));
        }
    }
}

__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -FILTER_WIDTH / 2; i <= FILTER_WIDTH / 2; i++) {
            for (int j = -FILTER_WIDTH / 2; j <= FILTER_WIDTH / 2; j++) {
                int ni = min(max(y + i, 0), height - 1);
                int nj = min(max(x + j, 0), width - 1);
                sum += Pin[ni * width + nj];
            }
        }
        Pout[y * width + x] = static_cast<unsigned char>(sum / (FILTER_WIDTH * FILTER_WIDTH));
    }
}

__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    __shared__ unsigned char sharedMem[TILE_SIZE + FILTER_WIDTH - 1][TILE_SIZE + FILTER_WIDTH - 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int lx = threadIdx.x + FILTER_WIDTH / 2;  // Local x index within shared memory
    int ly = threadIdx.y + FILTER_WIDTH / 2;  // Local y index within shared memory

    // Load data into shared memory
    if (x < width && y < height) {
        sharedMem[ly][lx] = Pin[y * width + x];
    } else {
        sharedMem[ly][lx] = 0;  // Pad boundary with 0
    }

    __syncthreads();

    if (x < width && y < height && threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        float sum = 0.0f;
        for (int i = -FILTER_WIDTH / 2; i <= FILTER_WIDTH / 2; i++) {
            for (int j = -FILTER_WIDTH / 2; j <= FILTER_WIDTH / 2; j++) {
                int si = ly + i;  // Shared memory index for y
                int sj = lx + j;  // Shared memory index for x

                if (si >= 0 && si < TILE_SIZE + FILTER_WIDTH - 1 && sj >= 0 && sj < TILE_SIZE + FILTER_WIDTH - 1) {
                    sum += sharedMem[si][sj];
                }
            }
        }
        Pout[y * width + x] = static_cast<unsigned char>(sum / (FILTER_WIDTH * FILTER_WIDTH));
    }
}


bool verify(cv::Mat &answer1, cv::Mat &answer2, unsigned int nRows, unsigned int nCols) {
    float tolerance = 10.0f;  // Allow some small difference (tune this value as needed)
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            if (abs(answer1.at<unsigned char>(i, j) - answer2.at<unsigned char>(i, j)) > tolerance) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << static_cast<int>(answer1.at<unsigned char>(i, j)) << " vs "
                          << static_cast<int>(answer2.at<unsigned char>(i, j)) << std::endl;
                return false;
            }
        }
    }
    return true;
}

void blurImage_d(cv::Mat &Pout_Mat_h, cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    unsigned char *d_Pout, *d_Pin;
    size_t size = nRows * nCols * sizeof(unsigned char);

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_Pout, size));
    CHECK(cudaMalloc((void**)&d_Pin, size));

    // Copy input image to device
    CHECK(cudaMemcpy(d_Pin, Pin_Mat_h.data, size, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((nCols + TILE_SIZE - 1) / TILE_SIZE, (nRows + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    blurImage_Kernel<<<numBlocks, threadsPerBlock>>>(d_Pout, d_Pin, nCols, nRows);
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(Pout_Mat_h.data, d_Pout, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_Pout));
    CHECK(cudaFree(d_Pin));
}

void blurImage_tiled_d(cv::Mat &Pout_Mat_h, cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    unsigned char *d_Pout, *d_Pin;
    size_t size = nRows * nCols * sizeof(unsigned char);

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_Pout, size));
    CHECK(cudaMalloc((void**)&d_Pin, size));

    // Copy input image to device
    CHECK(cudaMemcpy(d_Pin, Pin_Mat_h.data, size, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((nCols + TILE_SIZE - 1) / TILE_SIZE, (nRows + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    blurImage_tiled_Kernel<<<numBlocks, threadsPerBlock>>>(d_Pout, d_Pin, nCols, nRows);
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(Pout_Mat_h.data, d_Pout, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_Pout));
    CHECK(cudaFree(d_Pin));
}
