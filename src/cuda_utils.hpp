#ifndef CUPLYR_CUDA_UTILS_HPP
#define CUPLYR_CUDA_UTILS_HPP

#include <Rcpp.h>
#include <cuda_runtime.h>

namespace cuplyr {

inline void check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        Rcpp::stop("CUDA error (%s): %s", context, cudaGetErrorString(err));
    }
}

} // namespace cuplyr

#endif // CUPLYR_CUDA_UTILS_HPP
