#include <opencv2/core/cuda.hpp>

void loadExtractPatchFromImageKernel(const cv::cuda::GpuMat& img,
                                     float* d_patch, float* d_patchWithBorder,
                                     const int patchSize,
                                     const rovio::FeatureCoordinates& c);
