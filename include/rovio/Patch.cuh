#include <opencv2/core/cuda.hpp>

void loadExtractPatchFromImageKernel(const cv::cuda::GpuMat& img,
                                     float* d_patch,
                                     const int halfpatch_size,
                                     const rovio::FeatureCoordinates& c);
