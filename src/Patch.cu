#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev.hpp>

#include "rovio/FeatureCoordinates.hpp"
#include "rovio/Patch.cuh"

__global__ void extractPatchFromImageNearIdentityWarpingKernel(
    const cv::cudev::GlobPtrSz<const uchar> d_img, float* d_patch,
    const int halfpatch_size, const float c_x,
    const float c_y) {
  const int refStep = d_img.step;

  // Calculate global thread ID
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 2 * halfpatch_size || y >= 2 * halfpatch_size) return;

  const int u_r = floor(c_x);
  const int v_r = floor(c_y);

  // Compute interpolation weights
  const float subpix_x = c_x - u_r;
  const float subpix_y = c_y - v_r;
  const float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
  const float wTR = subpix_x * (1.0 - subpix_y);
  const float wBL = (1.0 - subpix_x) * subpix_y;
  const float wBR = subpix_x * subpix_y;

  const int img_x = u_r - halfpatch_size + x;
  const int img_y = v_r - halfpatch_size + y;
  if (img_x < 0 || img_y < 0 || img_x >= d_img.cols || img_y >= d_img.rows)
    return;

  const uchar* img_ptr = d_img.data + img_y * refStep + img_x;
  float* patch_ptr = d_patch + y * 2 * halfpatch_size + x;
  *patch_ptr = wTL * img_ptr[0];
  if (subpix_x > 0) *patch_ptr += wTR * img_ptr[1];
  if (subpix_y > 0) *patch_ptr += wBL * img_ptr[refStep];
  if (subpix_x > 0 && subpix_y > 0) *patch_ptr += wBR * img_ptr[refStep + 1];
}

__global__ void extractPatchFromImageKernel(
    const cv::cudev::GlobPtrSz<const uchar> d_img, float* d_patch,
    const int halfpatch_size, const float c_x,
    const float c_y,
    // const float warp_c[2][2]
    const float warp_c_00, const float warp_c_01,
    const float warp_c_10, const float warp_c_11
    ) {
  const int refStep = d_img.step;

  // Calculate global thread ID
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 2 * halfpatch_size || y >= 2 * halfpatch_size) return;

  const float dx = x - halfpatch_size + 0.5f;
  const float dy = y - halfpatch_size + 0.5f;

  // printf("warp_c: %f, %f, %f, %f\n", warp_c[0][0], warp_c[0][1], warp_c[1][0], warp_c[1][1]);
  // const float wdx = warp_c[0][0] * dx + warp_c[0][1] * dy;
  // const float wdy = warp_c[1][0] * dx + warp_c[1][1] * dy;
  const float wdx = warp_c_00 * dx + warp_c_01 * dy;
  const float wdy = warp_c_10 * dx + warp_c_11 * dy;

  const float u_pixel = c_x + wdx - 0.5f;
  const float v_pixel = c_y + wdy - 0.5f;

  const int u_r = floor(u_pixel);
  const int v_r = floor(v_pixel);
  if (u_r < 0 || v_r < 0 || u_r >= d_img.cols || v_r >= d_img.rows) return;
  // printf("u_r: %d, v_r: %d\n", u_r, v_r);

  const float subpix_x = u_pixel - u_r;
  const float subpix_y = v_pixel - v_r;

  const float wTL = (1.0f - subpix_x) * (1.0f - subpix_y);
  const float wTR = subpix_x * (1.0f - subpix_y);
  const float wBL = (1.0f - subpix_x) * subpix_y;
  const float wBR = subpix_x * subpix_y;

  const uchar* img_ptr = d_img.data + v_r * refStep + u_r;
  float* patch_ptr = d_patch + y * 2 * halfpatch_size + x;
  *patch_ptr = wTL * img_ptr[0];
  if (subpix_x > 0) *patch_ptr += wTR * img_ptr[1];
  if (subpix_y > 0) *patch_ptr += wBL * img_ptr[refStep];
  if (subpix_x > 0 && subpix_y > 0) *patch_ptr += wBR * img_ptr[refStep + 1];
}

void loadExtractPatchFromImageKernel(const cv::cuda::GpuMat& img,
                                     float* d_patch,
                                     const int halfpatch_size,
                                     const rovio::FeatureCoordinates& c) {
  const cv::cudev::GlobPtrSz<const uchar> d_img =
      cv::cudev::globPtr(img.ptr<uchar>(), img.step, img.rows, img.cols);

  if (c.isNearIdentityWarping()) {
    dim3 blockSize(16, 16);  // Adjust as needed
    dim3 gridSize((2 * halfpatch_size + blockSize.x - 1) / blockSize.x,
                  (2 * halfpatch_size + blockSize.y - 1) / blockSize.y);
    extractPatchFromImageNearIdentityWarpingKernel<<<gridSize, blockSize>>>(
        d_img, d_patch, halfpatch_size, c.get_c().x, c.get_c().y);
  } else {
    // Convert Eigen::Matrix2f to a simple 2x2 float array
    float warp_c_array[2][2] = {{c.get_warp_c()(0, 0), c.get_warp_c()(0, 1)},
                                {c.get_warp_c()(1, 0), c.get_warp_c()(1, 1)}};

    dim3 blockSize(16, 16);  // Adjust as needed
    dim3 gridSize((2 * halfpatch_size + blockSize.x - 1) / blockSize.x,
                  (2 * halfpatch_size + blockSize.y - 1) / blockSize.y);
    extractPatchFromImageKernel<<<gridSize, blockSize>>>(
        d_img, d_patch, halfpatch_size, c.get_c().x, c.get_c().y,
        c.get_warp_c()(0, 0), c.get_warp_c()(0, 1), c.get_warp_c()(1, 0), c.get_warp_c()(1, 1));
        // warp_c_array);
  }

  cudaDeviceSynchronize();  // Wait for the kernel to complete

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("extractPatchFromImageKernel() failed: %s\n",
           cudaGetErrorString(error));
    exit(-1);
  }
}
